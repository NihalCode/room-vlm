"""API tests with mocked model service (no 4B download)."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Ensure app lifespan skips model load
os.environ["ROOM_VLM_SKIP_MODEL_LOAD"] = "1"

import room_vlm.api.app as api_module
from room_vlm.api.app import app


@pytest.fixture()
def client(monkeypatch):
    class FakeService:
        loaded = True
        model_name = "fake-model"
        adapter_name = "fake-adapter"

        def info(self):
            return {
                "model": self.model_name,
                "adapter": self.adapter_name,
                "device": "cpu",
                "labels": ["bedroom", "living room", "bathroom", "kitchen"],
                "strategy": "candidate_scoring",
            }

        def predict_image_file(self, path: Path):
            return {
                "label": "bedroom",
                "confidence": 0.91,
                "scores": {
                    "bedroom": 0.91,
                    "living room": 0.05,
                    "bathroom": 0.02,
                    "kitchen": 0.02,
                },
                "model": self.model_name,
                "adapter": self.adapter_name,
                "latency_ms": 12.3,
                "abstained": False,
            }

        def predict_video_file(self, path: Path, include_temporal: bool = False):
            payload = {
                "label": "kitchen",
                "confidence": 0.8,
                "scores": {
                    "bedroom": 0.05,
                    "living room": 0.05,
                    "bathroom": 0.1,
                    "kitchen": 0.8,
                },
                "model": self.model_name,
                "adapter": self.adapter_name,
                "latency_ms": 40.0,
                "abstained": False,
                "frames_used": 3,
                "temporal_predictions": None,
            }
            if include_temporal:
                payload["temporal_predictions"] = [{"label": "kitchen", "confidence": 0.8}]
            return payload

    monkeypatch.setattr(api_module, "model_service", FakeService())
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_model_info(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "fake-model"
    assert body["adapter"] == "fake-adapter"
    assert ":\\" not in (body.get("adapter") or "")


def test_predict_image(client, tmp_path):
    img_path = tmp_path / "room.jpg"
    Image.new("RGB", (32, 32), (100, 100, 100)).save(img_path)
    with img_path.open("rb") as f:
        resp = client.post("/predict/image", files={"file": ("room.jpg", f, "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "bedroom"
    assert "scores" in body
    assert "latency_ms" in body
