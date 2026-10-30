#!/usr/bin/env bash
# Build artifacts for Namioto. All output goes to dist/, work files to build/.
#
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

cd "$(dirname "$0")"

usage() {
    cat <<'EOF'
Usage: ./build.sh <target>

  wheel            sdist + wheel via uv build          -> dist/*.whl, dist/*.tar.gz
  cli              frozen task CLI via PyInstaller      -> dist/namioto-cli/
  ctl              frozen IPC client via PyInstaller    -> dist/namioto-ctl/
  app              frozen GUI via PyInstaller           -> dist/namioto/
  all              wheel + cli + ctl + app
  clean            remove build/ and dist/
EOF
}

pyinstaller_common=(
    --noconfirm
    --clean
    --specpath build
)

build_wheel() {
    uv build
}

build_cli() {
    uv sync --extra cpu
    freeze namioto-cli namioto.cli \
        --collect-submodules namioto.analysis
}

freeze() {
    # The entry script lives outside the package: PyInstaller puts the directory it infers from the
    # script ahead of site-packages, so a script inside `namioto/` would be answered by the source
    # tree instead of the installed one. The launcher also runs `freeze_support()` before the
    # program starts, which a frozen `spawn` child needs to find its way back into the parent.
    local name="$1" module="$2" launcher="build/$1-entry.py"
    shift 2
    mkdir -p build
    printf 'import multiprocessing\n\nmultiprocessing.freeze_support()\n\nfrom %s import main\n\nraise SystemExit(main())\n' \
        "$module" > "$launcher"
    uv run --no-editable --with pyinstaller pyinstaller "${pyinstaller_common[@]}" \
        --name "$name" \
        --onedir \
        "$@" \
        "$launcher"
    copy_licenses "dist/$name"
}

build_app() {
    # a frozen app collects what is installed, so the project is installed as a wheel rather than in
    # the editable layout, which would leave PyInstaller following a path hook into the checkout; the
    # reinstall is what picks up source edits, since the version itself does not change
    uv sync --locked --extra cpu --no-editable --reinstall-package namioto
    freeze namioto namioto.ui.app \
        --windowed \
        --collect-data qtawesome \
        --collect-data _sounddevice_data
}

copy_licenses() {
    cp LICENSE NOTICE "$1"
}

case "${1:-}" in
    wheel) build_wheel ;;
    cli) build_cli ;;
    ctl) freeze namioto-ctl namioto.ipc ;;
    app) build_app ;;
    all)
        build_wheel
        build_cli
        freeze namioto-ctl namioto.ipc
        build_app
        ;;
    clean) rm -rf build dist ;;
    *)
        usage
        exit 1
        ;;
esac