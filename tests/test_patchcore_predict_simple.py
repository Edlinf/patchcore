import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
INDAD = ROOT / "indad"
if str(INDAD) not in sys.path:
    sys.path.insert(0, str(INDAD))

from patchcore_predict_simple import collect_images, load_patchcore_archive_simple, parse_pair
from models import save_patchcore_archive, save_tensor


def test_parse_pair_reads_width_height():
    assert parse_pair("1280,128") == [1280, 128]
    assert parse_pair("48x16") == [48, 16]


def test_collect_images_reads_flat_directory_only(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.jpg").write_bytes(b"x")

    images = collect_images(tmp_path)

    assert [p.name for p in images] == ["a.jpg", "b.png"]


def test_load_patchcore_archive_simple_supports_old_archive(tmp_path):
    patch_lib = torch.randn(2, 3, 4, 5)
    save_tensor(str(tmp_path), "old.ts", patch_lib)

    loaded, stats = load_patchcore_archive_simple(tmp_path / "old.ts")

    assert torch.allclose(loaded, patch_lib)
    assert stats is None


def test_load_patchcore_archive_simple_supports_new_archive(tmp_path):
    patch_lib = torch.randn(2, 3, 4, 5)
    stats = {
        "baseline": torch.ones(2, 3),
        "scale": torch.full((2, 3), 2.0),
        "recommended_pixel_threshold": torch.tensor(3.5),
    }
    save_patchcore_archive(str(tmp_path), "new.ts", patch_lib, stats)

    loaded, loaded_stats = load_patchcore_archive_simple(tmp_path / "new.ts")

    assert torch.allclose(loaded, patch_lib)
    assert torch.allclose(loaded_stats["baseline"], stats["baseline"])
    assert torch.allclose(loaded_stats["scale"], stats["scale"])
