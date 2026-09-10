"""Inference package."""

from room_vlm.inference.image import predict_image
from room_vlm.inference.video import predict_video

__all__ = ["predict_image", "predict_video"]
