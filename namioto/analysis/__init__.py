# SPDX-License-Identifier: AGPL-3.0-only
"""Audio analysis and model inference, each module also a CLI.

Deliberately imports nothing: `namioto.settings` reads `choices` and `devices` for its defaults
while `transcription` reads `settings` back, so re-exporting here would close the loop.
"""
