"""API package.

Note: do not bind ``app`` at import time in a way that shadows the
``room_vlm.api.app`` submodule (needed for tests and uvicorn paths).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__ = ["app"]


def __getattr__(name: str):
    if name == "app":
        from room_vlm.api.app import app as fastapi_app

        return fastapi_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
