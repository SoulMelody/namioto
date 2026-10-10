#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Convert a FA-Kara forced-alignment model to the ONNX file and dictionary `namioto-cli align` loads.

A development tool, not a runtime dependency: it needs torch and, per kind, torchaudio or
transformers, plus ONNX, none of which the program itself imports. It writes `model.onnx` and
`vocab.json` into the aligner's data directory, or into `--out`. `--precision int8` (the default)
quantises the MatMul weights, which is what the CPU wants; `fp16` writes a `model.fp16.onnx` beside
it, which is what a GPU run loads; `fp32` leaves the graph as exported. The WebGPU provider cannot
run quantised matmuls whole - it falls back to the CPU and copies their activations across for each
one, which costs more than the matmul did.

    uv run --group export scripts/export_align_model.py --model mms
    uv run --group export scripts/export_align_model.py --model yohane

`mms` is Meta's MMS forced-alignment checkpoint (`torchaudio.pipelines.MMS_FA`), the default;
`yohane` is `NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn`, a karaoke fine-tune. Both are
wav2vec2 CTC models trained on romanised kana. The graph bakes in the waveform normalisation
the models expect, takes `input_values` and returns `logits`; quantisation is limited to `MatMul`,
because the exporter writes the convolutions' bias outside the initializers and quantising those
fails. The attention's `where(isnan(softmax), 0, softmax)` guard is dropped: it is a dead branch for
the finite input the graph is handed, and the graph is smaller without it. So is the attention mask
the HuggingFace models put on every layer: a window here is exactly what the model attends over, so
that mask is all zeros, and the add it feeds is quadratic in the frame count.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

from namioto.analysis import align

HF_MODELS = {"yohane": "NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn"}
SECONDS = 16
OPSET = 18


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=align.MODELS, default=align.DEFAULT_MODEL)
    parser.add_argument("--language", choices=align.LANGUAGES, default="ja")
    parser.add_argument(
        "--hf-model", help=f"a HuggingFace model id for --model yohane (default: {HF_MODELS['yohane']})"
    )
    parser.add_argument("--out", type=pathlib.Path, help="where to write it (default: the data directory)")
    parser.add_argument(
        "--precision",
        choices=("int8", "fp16", "fp32"),
        default="int8",
        help="int8 (default) quantises the MatMul weights for the CPU; fp16 halves their width for a "
        "GPU run; fp32 keeps the exported graph",
    )
    return parser.parse_args(argv)


def drop_nan_guards(graph) -> int:
    """Drop every `where(isnan(x), fallback, x)` guard in `graph` and return how many were removed."""
    guard = {node.output[0]: node for node in graph.node if node.op_type == "IsNaN"}
    replace, dropped = {}, set()
    for consumer in graph.node:
        if consumer.op_type != "Where" or consumer.input[0] not in guard:
            continue
        guarded = guard[consumer.input[0]].input[0]
        if consumer.input[2] == guarded:
            replace[consumer.output[0]] = guarded
            dropped.update({guard[consumer.input[0]].name, consumer.name})
    if not dropped:
        return 0
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name in replace:
                node.input[index] = replace[name]
    kept = [node for node in graph.node if node.name not in dropped]
    graph.ClearField("node")
    graph.node.extend(kept)
    return len(dropped)


def drop_attention_mask(graph) -> int:
    """Drop the per-layer attention mask of `graph` and return how many of them were removed.

    `transformers` hands every layer a bidirectional mask: a `where` builds a
    `[batch, 1, frames, frames]` bias over the scores and an `add` puts it on. Every window this
    program hands the model is exactly the frames the model attends over, so the bias is all zeros
    and the add is an identity - but it is quadratic in the frame count, and a GPU spends on it more
    than on the whole encoder. The add goes, and whatever only built the mask with it.
    """
    masks = {node.output[0] for node in graph.node if node.op_type == "Where"}
    replace, dropped = {}, set()
    for node in graph.node:
        if node.op_type != "Add" or "/attention/" not in node.name:
            continue
        for index, name in enumerate(node.input):
            if name in masks:
                replace[node.output[0]] = node.input[1 - index]
                dropped.add(node.name)
    if not dropped:
        return 0
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name in replace:
                node.input[index] = replace[name]
    nodes = list(graph.node)
    producers = {name: index for index, node in enumerate(nodes) for name in node.output}
    needed, frontier = set(), [output.name for output in graph.output]
    while frontier:
        index = producers.get(frontier.pop())
        if index is None or index in needed:
            continue
        needed.add(index)
        frontier.extend(nodes[index].input)
    kept = [node for index, node in enumerate(nodes) if index in needed]
    graph.ClearField("node")
    graph.node.extend(kept)
    return len(dropped)


def export(module, destination: pathlib.Path, precision: str = "int8") -> None:
    """Write the ONNX graph of `module`, which takes raw audio and returns `logits`.

    `precision` is `int8` for a dynamic quantisation of the MatMul weights, `fp16` for the graph with
    half-precision weights, or `fp32` for the graph as it was exported.
    """
    import onnx
    import torch

    with tempfile.TemporaryDirectory() as scratch:
        source = pathlib.Path(scratch) / "fp32.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                module,
                (torch.randn(1, SECONDS * align.SAMPLE_RATE),),
                source,
                input_names=["input_values"],
                output_names=["logits"],
                dynamic_axes={
                    "input_values": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"},
                },
                opset_version=OPSET,
                do_constant_folding=True,
                # the dynamo exporter ignores dynamic_axes, which the variable-length windows need
                dynamo=False,
            )
        model = onnx.load(str(source))
        guards = drop_nan_guards(model.graph)
        masks = drop_attention_mask(model.graph)
        print(f"dropped {guards} nan guard node(s) and {masks} attention mask node(s)")
        if precision == "fp32":
            onnx.save(model, str(destination))
            return
        if precision == "fp16":
            from onnxruntime.transformers.float16 import convert_float_to_float16

            onnx.save(convert_float_to_float16(model, keep_io_types=True), str(destination))
            return
        onnx.save(model, str(source))
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(source), str(destination), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])


def export_mms(destination: pathlib.Path, precision: str = "int8") -> dict[str, int]:
    """Meta's MMS checkpoint: layer norm folded in, log-softmax and the star token left off."""
    import torch
    import torchaudio

    model = torchaudio.pipelines.MMS_FA.get_model(with_star=False)

    class Mms(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, values):
            values = (values - values.mean()) / torch.sqrt(values.var(unbiased=False) + 1e-5)
            logits, _ = self.inner(values)
            return logits

    # the wrapper's own layer_norm takes its shape from the waveform, which the exporter cannot
    # bake into the graph; the manual normalisation above replaces it
    model.normalize_waveform = False
    model.apply_log_softmax = False
    export(Mms(model), destination, precision)

    labels = torchaudio.pipelines.MMS_FA.get_labels(star=None)
    return {"[pad]": 0, **{label: index for index, label in enumerate(labels) if index}}


def export_yohane(destination: pathlib.Path, name: str, precision: str = "int8") -> dict[str, int]:
    """The karaoke fine-tune: the feature extractor's layer norm folded into the graph."""
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(name)
    model = Wav2Vec2ForCTC.from_pretrained(name).eval()

    class Yohane(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, values):
            values = (values - values.mean()) / torch.sqrt(values.var(unbiased=False) + 1e-7)
            return self.inner(values).logits

    export(Yohane(model), destination, precision)
    return {char.lower(): int(code) for char, code in processor.tokenizer.get_vocab().items()}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    destination = args.out or align.model_dir(args.model, args.language)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / (align.FP16_FILE if args.precision == "fp16" else align.MODEL_FILE)
    if args.model == "yohane":
        vocab = export_yohane(target, args.hf_model or HF_MODELS["yohane"], args.precision)
    else:
        vocab = export_mms(target, args.precision)
    (destination / align.VOCAB_FILE).write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {target} and {destination / align.VOCAB_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
