"""Compatibility helpers for direct-reference configuration loading."""

from __future__ import annotations

from pathlib import Path

from .config import DirectDecodingProfile
from .protocol import load_direct_reference_protocol


def load_decoding_profiles(path: Path) -> dict[str, DirectDecodingProfile]:
    protocol = load_direct_reference_protocol(path)
    return {profile.profile_id: profile for profile in protocol.decoding_profiles}
