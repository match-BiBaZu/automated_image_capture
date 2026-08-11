from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

SUPPORTED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})

CAPTURE_NAME = re.compile(
    r"^img_(?P<index>\d+)_(?:ur(?P<pose>\d+)|ura-(?P<angle>\d+))_"
    r"(?:belt-(?P<belt>\d+)_pos-(?P<position>\d+)_(?P<direction>out|back)_)?"
    r"(?:ramp-(?P<ramp>\d+)_)?"
    r"p1-(?P<p1>\d+)_p2-(?P<p2>\d+)_"
    r"(?P<exposure>auto|e\d+us)(?P<suffix>\.png|\.jpe?g|\.webp)$",
    re.IGNORECASE,
)


def is_supported_image(path: Path) -> bool:
    return path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES


def iter_images(directory: Path, *, recursive: bool = False) -> Iterator[Path]:
    candidates = directory.rglob("*") if recursive else directory.glob("*")
    for path in candidates:
        if path.is_file() and is_supported_image(path):
            yield path
