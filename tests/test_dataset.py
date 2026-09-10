"""Dataset loading tests using fixtures."""

from pathlib import Path

from room_vlm.data.dataset import RoomImageDataset, read_jsonl, write_jsonl, ExampleRecord
from room_vlm.nyu.discovery import discover_scenes
from room_vlm.nyu.sampling import sample_scene_frames

FIXTURES = Path(__file__).parent / "fixtures" / "nyu"


def test_dataset_loads_image_and_label(tmp_path: Path):
    scenes = discover_scenes(FIXTURES)
    scene = scenes[0]
    _, sampled = sample_scene_frames(scene.path, strategy="every_n_frames", every_n_frames=1)
    assert sampled
    frame = sampled[0]
    records = [
        ExampleRecord(
            image=str(frame.path),
            label=scene.label,
            scene_id=scene.scene_id,
        )
    ]
    path = tmp_path / "train.jsonl"
    write_jsonl(records, path)
    rows = read_jsonl(path)
    ds = RoomImageDataset(rows)
    item = ds[0]
    assert item["label"] == scene.label
    assert item["scene_id"] == scene.scene_id
    assert item["image"].size[0] > 0
