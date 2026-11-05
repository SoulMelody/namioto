#!/usr/bin/env bash
# Publish the ONNX models the program downloads on demand as the assets of one release, so
# namioto/model_store.py can fetch them. Nothing here touches git: gh creates the tag on the remote.
#
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

cd "$(dirname "$0")/.."

# Defaults for this project's release; every one of them can be overridden on the command line.
REPO="SoulMelody/namioto"
TAG="models"
TITLE="ONNX models"
NOTES="ONNX models, downloaded on demand by namioto.

- tempocnn-onnx.zip — Essentia/TempoCNN deeptemp-k16-3 (CC BY-NC-SA 4.0)

It carries a non-commercial license: it is redistributed for use with namioto only. See NOTICE."
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/namioto/models"
TEMPOCNN=""
OUT="dist/models"
ONLY="tempocnn"
REUSE=0
CLOBBER=0
DRY=0

usage() {
    cat <<'EOF'
Usage: ./scripts/upload_models.sh [options]

Publishes tempocnn-onnx.zip as the asset of one GitHub release and prints the registry line that
points the program at it.
`gh` must be logged in with a token that carries the `repo` scope for the repository's owner.

  --tag TAG          the release tag (default: models). A re-export wants a new one: the program
                     caches a model by its path, so replacing an asset under a published tag would
                     change results under it
  --only tempocnn    which models to publish (default: all of them)
  --repo OWNER/NAME  default: SoulMelody/namioto
  --data-dir DIR     where the models live (default: $XDG_DATA_HOME/namioto/models)
  --tempocnn FILE    the TempoCNN .onnx (default: the one in the data directory)
  --out DIR          where the zips are built (default: dist/models)
  --title TEXT       release title (default: "ONNX models")
  --notes TEXT       release notes (default: attribution and license summary)
  --reuse            upload into an existing release instead of refusing it; an asset that release
                     already holds is left as it is, so a package can be added beside it
  --clobber          replace assets the release already holds, instead of leaving them as they are
                     (breaks a published download)
  --dry-run          print what would run, touch nothing
EOF
}

die() {
    echo "$1" >&2
    exit 1
}

say() {
    echo "==> $1"
}

run() {
    if [ "$DRY" == 1 ]; then
        printf '    would run: %s\n' "$*"
    else
        "$@"
    fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        --tag)
            TAG="${2:?--tag needs a value}"
            shift
            ;;
        --repo)
            REPO="${2:?--repo needs a value}"
            shift
            ;;
        --data-dir)
            DATA_DIR="${2:?--data-dir needs a value}"
            shift
            ;;
        --tempocnn)
            TEMPOCNN="${2:?--tempocnn needs a value}"
            shift
            ;;
        --out)
            OUT="${2:?--out needs a value}"
            shift
            ;;
        --title)
            TITLE="${2:?--title needs a value}"
            shift
            ;;
        --notes)
            NOTES="${2:?--notes needs a value}"
            shift
            ;;
        --only)
            ONLY="${2//,/ }"
            shift
            ;;
        --reuse) REUSE=1 ;;
        --clobber) CLOBBER=1 ;;
        --dry-run) DRY=1 ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            die "Unknown option: $1"
            ;;
    esac
    shift
done

[ -n "$REPO" ] || die "--repo cannot be empty"
[ -n "$TAG" ] || die "--tag cannot be empty"

command -v zip >/dev/null || die "zip is needed and is not installed"
command -v sha256sum >/dev/null || die "sha256sum is needed and is not installed"
if [ "$DRY" == 0 ]; then
    command -v gh >/dev/null || die "gh is needed and is not installed"
    # the active account, not `gh auth status`: that one also fails on another account's dead token
    gh api user >/dev/null 2>&1 || die "gh is not logged in; run: gh auth login"
    gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1 && {
        [ "$REUSE" == 1 ] || die "release $TAG already exists; a re-export wants a new tag, or --reuse to add to it"
    } || true
fi

# One zip per ONNX model, flat, and named the way GAME's own release assets are: the model, then
# -onnx. The language is not in the name, so a second language for one model would need its own
# tag.
assets=()
build_zip() {
    local asset="$1"
    shift
    for source in "$@"; do
        [ -f "$source" ] || die "$source is missing"
    done
    say "packing $asset"
    run rm -f "$OUT/$asset"
    run zip -j -q "$OUT/$asset" "$@"
    assets+=("$OUT/$asset")
}

if [ "$DRY" == 0 ]; then
    mkdir -p "$OUT"
fi
for model in $ONLY; do
    case "$model" in
        tempocnn)
            source="$TEMPOCNN"
            [ -n "$source" ] || source="$DATA_DIR/tempocnn/deeptemp-k16-3.onnx"
            license="LICENSE-CC-BY-NC-SA-4.0.txt"
            if [ -f "$license" ]; then
                build_zip "tempocnn-onnx.zip" "$source" "$license"
            else
                build_zip "tempocnn-onnx.zip" "$source"
            fi
            ;;
        *)
            die "Unknown model: $model (expected tempocnn)"
            ;;
    esac
done

say "recording the digests"
if [ "$DRY" == 0 ]; then
    (cd "$OUT" && sha256sum ./*.zip >SHA256SUMS)
    cat "$OUT/SHA256SUMS"
fi

say "publishing release $TAG on $REPO"
if [ "$DRY" == 1 ]; then
    printf '    would run: gh release create %s --repo %s --title %s --notes-file - --latest=false\n' \
        "$TAG" "$REPO" "$TITLE"
else
    if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
        say "release $TAG is already there, adding to it"
    else
        printf '%s\n' "$NOTES" | gh release create "$TAG" --repo "$REPO" --title "$TITLE" --notes-file - --latest=false
    fi
fi

upload=(gh release upload "$TAG" --repo "$REPO")
if [ "$CLOBBER" == 1 ]; then
    upload+=(--clobber)
fi

# an asset the release already holds is left alone, so a package can be added beside the ones it was
# published with; --clobber is how a published one is replaced instead
published=""
if [ "$DRY" == 0 ]; then
    published="$(gh api "repos/$REPO/releases/tags/$TAG" --jq '.assets[].name')"
fi
fresh=()
for asset in "${assets[@]}"; do
    name="$(basename "$asset")"
    if [ "$CLOBBER" == 0 ] && printf '%s\n' "$published" | grep -qxF -- "$name"; then
        say "$TAG already holds $name; leaving it as it is"
    else
        fresh+=("$asset")
    fi
done
if [ "${#fresh[@]}" -gt 0 ]; then
    run "${upload[@]}" "${fresh[@]}"
fi

registry_hint() {
    cat <<EOF

Point the registry at the release (namioto/model_store.py, MODELS):

  "tempocnn": Model(
      name="tempocnn",
      env="NAMIOTO_TEMPOCNN_MODEL",
      files=("deeptemp-k16-3.onnx",),
      asset="https://github.com/$REPO/releases/download/$TAG/tempocnn-onnx.zip",
  ),

model_store.Model.url() fills {model} with the first key of the entry - the size for game - so the
asset that has no key at all is written with none.
EOF
}

if [ "$DRY" == 1 ]; then
    say "dry run: nothing was published"
    registry_hint
    exit 0
fi

say "checking what the release holds"
gh api "repos/$REPO/releases/tags/$TAG" \
    --jq '.assets[] | "\(.name)\t\(.size)\t\(.browser_download_url)"' |
    tee "$OUT/assets.txt"

for asset in "${fresh[@]}"; do
    name="$(basename "$asset")"
    local_size="$(stat -c%s "$asset")"
    remote_size="$(awk -v name="$name" '$1 == name { print $2 }' "$OUT/assets.txt")"
    [ "$remote_size" == "$local_size" ] || die "$name is $remote_size bytes on the release, not $local_size"
done
if [ "${#fresh[@]}" -gt 0 ]; then
    say "every asset this run uploaded matches its local zip"
else
    say "nothing to upload; the release already holds every asset"
fi
registry_hint
