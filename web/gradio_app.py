#!/usr/bin/env python3
"""Simple Gradio UI for image/video room classification."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr

from room_vlm.inference.image import predict_image
from room_vlm.inference.video import predict_video
from room_vlm.model.loader import attach_adapter, load_model_and_processor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("gradio_app")

_MODEL = None
_PROCESSOR = None
_DEVICE = None


def get_model():
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is not None:
        return _MODEL, _PROCESSOR, _DEVICE
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-4B-Instruct")
    adapter = os.environ.get("ADAPTER_PATH") or None
    device = os.environ.get("DEVICE", "auto")
    loaded = load_model_and_processor(
        model_name,
        quantization={"enabled": False},
        device=device,
        allow_bf16_lora=True,
    )
    model = loaded.model
    if adapter:
        model = attach_adapter(model, adapter)
    model.eval()
    _MODEL, _PROCESSOR, _DEVICE = model, loaded.processor, loaded.device
    return _MODEL, _PROCESSOR, _DEVICE


def run_image(image):
    if image is None:
        return "No image", "", ""
    model, processor, device = get_model()
    result = predict_image(model, processor, image, device=device)
    scores = "\n".join(f"{k}: {v:.3f}" for k, v in result["scores"].items())
    return result["label"], f"{result['confidence']:.3f}", scores


def run_video(video):
    if video is None:
        return "No video", "", "", ""
    model, processor, device = get_model()
    max_frames = int(os.environ.get("MAX_VIDEO_FRAMES", "32"))
    result = predict_video(
        model,
        processor,
        video,
        max_frames=max_frames,
        include_temporal=True,
        device=device,
    )
    scores = "\n".join(f"{k}: {v:.3f}" for k, v in result["scores"].items())
    temporal = "\n".join(
        f"t={t.get('timestamp'):.1f}s -> {t.get('label')} ({t.get('confidence'):.2f})"
        for t in (result.get("temporal_predictions") or [])
    )
    return (
        result["prediction"],
        f"{result['confidence']:.3f}",
        str(result.get("frames_used")),
        scores + "\n\n" + temporal,
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Room VLM") as demo:
        gr.Markdown("# Room VLM\nIndoor room classification (bedroom / living room / bathroom / kitchen)")
        with gr.Tab("Image"):
            image = gr.Image(type="pil", label="Upload image")
            btn_i = gr.Button("Predict")
            out_label = gr.Textbox(label="Predicted room")
            out_conf = gr.Textbox(label="Confidence")
            out_scores = gr.Textbox(label="Class scores")
            btn_i.click(run_image, inputs=[image], outputs=[out_label, out_conf, out_scores])
        with gr.Tab("Video"):
            video = gr.Video(label="Upload video")
            btn_v = gr.Button("Predict")
            v_label = gr.Textbox(label="Predicted room")
            v_conf = gr.Textbox(label="Confidence")
            v_frames = gr.Textbox(label="Frames sampled")
            v_detail = gr.Textbox(label="Scores / temporal")
            btn_v.click(run_video, inputs=[video], outputs=[v_label, v_conf, v_frames, v_detail])
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
