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
    --paths .
    --specpath build
)

build_wheel() {
    uv build
}

build_cli() {
    uv sync --extra cpu
    freeze_cli namioto/cli.py namioto-cli \
        --collect-submodules namioto.analysis
}

freeze_cli() {
    local entry="$1" name="$2"
    shift 2
    uv run --no-editable --with pyinstaller pyinstaller "${pyinstaller_common[@]}" \
        --name "$name" \
        --onedir \
        "$@" \
        "$entry"
    copy_licenses "dist/$name"
}

build_app() {
    # a frozen app needs the compiled extension inside the package, so the project is installed as
    # a wheel rather than in the editable layout, which keeps it under build/
    uv sync --extra cpu --no-editable --reinstall-package namioto
    uv run --no-editable --with pyinstaller pyinstaller "${pyinstaller_common[@]}" \
        --name namioto \
        --onedir \
        --windowed \
        --collect-data qtawesome \
        namioto/ui/app.py
    copy_licenses dist/namioto
}

copy_licenses() {
    cp LICENSE NOTICE "$1"
}

case "${1:-}" in
    wheel) build_wheel ;;
    cli) build_cli ;;
    ctl) freeze_cli namioto/ipc.py namioto-ctl ;;
    app) build_app ;;
    all)
        build_wheel
        build_cli
        freeze_cli namioto/ipc.py namioto-ctl
        build_app
        ;;
    clean) rm -rf build dist ;;
    *)
        usage
        exit 1
        ;;
esac
