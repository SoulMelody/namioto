#!/usr/bin/env bash
# Rewrite the package-index URLs uv recorded in uv.lock back to their upstream
# hosts. `uv lock` takes the index from the developer's uv.toml, so a local
# mirror lands in the lockfile and CI - which has no mirror - sees a stale lock.
#
# Run from anywhere; the lockfile is the one beside this repository root.
#
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

lock="$(cd "$(dirname "$0")/.." && pwd)/uv.lock"

# index -> default PyPI, wheels/sdists -> pythonhosted.org, where the mirror
# serves the same files under the same paths and hashes
sed -i \
    -e "s|https://pypi.tuna.tsinghua.edu.cn/simple|https://pypi.org/simple|g" \
    -e "s|https://pypi.tuna.tsinghua.edu.cn/packages|https://files.pythonhosted.org/packages|g" \
    -e "s|https://mirrors.cloud.tencent.com/pypi/simple|https://pypi.org/simple|g" \
    -e "s|https://mirrors.cloud.tencent.com/pypi/packages|https://files.pythonhosted.org/packages|g" \
    -e "s|https://mirrors.aliyun.com/pypi/simple|https://pypi.org/simple|g" \
    -e "s|https://mirrors.aliyun.com/pypi/packages|https://files.pythonhosted.org/packages|g" \
    "$lock"

echo "patched $lock"