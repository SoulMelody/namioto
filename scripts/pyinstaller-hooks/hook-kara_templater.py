# SPDX-License-Identifier: AGPL-3.0-only
"""Collect native metrics, Windows wheel DLLs and package license metadata."""

from PyInstaller.utils.hooks import collect_all, collect_delvewheel_libs_directory, copy_metadata

datas, binaries, hiddenimports = collect_all("kara_templater")
datas, binaries = collect_delvewheel_libs_directory("kara_templater", datas=datas, binaries=binaries)
datas += copy_metadata("kara-templater")
