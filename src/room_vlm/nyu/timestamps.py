"""Parse timestamps from NYU Depth V2 raw filenames."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Example: r-1294886362.238178-3118787619.ppm
NYU_TIMESTAMP_RE = re.compile(
    r"^[rda]-(?P<ts>\d+\.\d+)-(?P<seq>\d+)\.(?P<ext>\w+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TimestampParseResult:
    timestamp: float | None
    sequence: str | None
    ok: bool
    fallback_used: bool
    message: str | None = None


def parse_timestamp_from_filename(filename: str) -> TimestampParseResult:
    """
    Extract the floating-point timestamp from an NYU raw filename.

    Returns a structured result. Callers should use
    :func:`timestamp_or_fallback` when a numeric value is required.
    """
    name = filename.strip().split("/")[-1].split("\\")[-1]
    match = NYU_TIMESTAMP_RE.match(name)
    if not match:
        return TimestampParseResult(
            timestamp=None,
            sequence=None,
            ok=False,
            fallback_used=False,
            message=f"Could not parse NYU timestamp from filename: {filename}",
        )
    try:
        ts = float(match.group("ts"))
    except ValueError:
        return TimestampParseResult(
            timestamp=None,
            sequence=match.group("seq"),
            ok=False,
            fallback_used=False,
            message=f"Invalid timestamp float in filename: {filename}",
        )
    return TimestampParseResult(
        timestamp=ts,
        sequence=match.group("seq"),
        ok=True,
        fallback_used=False,
        message=None,
    )


def timestamp_or_fallback(filename: str, frame_index: int) -> tuple[float, bool]:
    """
    Return (timestamp, used_fallback).

    If parsing fails, log a warning and return a deterministic synthetic
    timestamp based on frame_index (seconds). Frames are never silently discarded.
    """
    result = parse_timestamp_from_filename(filename)
    if result.ok and result.timestamp is not None:
        return result.timestamp, False
    logger.warning(
        "%s; using synthetic timestamp from frame_index=%d",
        result.message or "timestamp parse failed",
        frame_index,
    )
    return float(frame_index), True
