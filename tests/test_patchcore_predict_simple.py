import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
INDAD = ROOT / "indad"
if str(INDAD) not in sys.path:
    sys.path.insert(0, str(INDAD))

from patchcore_predict_simple import collect_images, infer_label_from_path, load_patchcore_archive_simple, parse_pair
from models import save_patchcore_archive, save_tensor


def test_parse_pair_reads_width_height():
    assert parse_pair("1280,128") == [1280, 128]
    assert parse_pair("48x16") == [48, 16]


def test_collect_images_reads_flat_directory_or_labeled_children(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.jpg").write_bytes(b"x")

    images = collect_images(tmp_path)

    assert [p.name for p in images] == ["a.jpg", "b.png"]

    root = tmp_path / "labeled"
    (root / "good").mkdir(parents=True)
    (root / "bad").mkdir(parents=True)
    (root / "good" / "ok.jpg").write_bytes(b"x")
    (root / "bad" / "ng.jpg").write_bytes(b"x")

    labeled_images = collect_images(root)

    assert [p.name for p in labeled_images] == ["ng.jpg", "ok.jpg"]


def test_infer_label_from_path_uses_parent_directory():
    assert infer_label_from_path(Path("D:/data/test/good/a.jpg")) == 0
    assert infer_label_from_path(Path("D:/data/test/OK/a.jpg")) == 0
    assert infer_label_from_path(Path("D:/data/test/bad/a.jpg")) == 1
    assert infer_label_from_path(Path("D:/data/test/NG/a.jpg")) == 1
    assert infer_label_from_path(Path("D:/data/test/unknown/a.jpg")) == -1


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


from patchcore_predict_simple import apply_score_stats, parse_model_info_simple


def test_apply_score_stats_clamps_negative_values():
    raw = torch.tensor([[1.0, 5.0]])
    stats = {
        "baseline": torch.tensor([[2.0, 1.0]]),
        "scale": torch.tensor([[1.0, 2.0]]),
    }

    out = apply_score_stats(raw, stats)

    assert torch.allclose(out, torch.tensor([[0.0, 2.0]]))


def test_parse_model_info_simple_reads_current_filename():
    info = parse_model_info_simple("patchcore_job_cv2_resnet18_23_16x48_3x128x384_fp32_abcd1234.ts")

    assert info["method"] == "patchcore"
    assert info["resize_method"] == "cv2"
    assert info["backbone"] == "resnet18"
    assert info["out_indices"] == [2, 3]
    assert info["fmap_size"] == [16, 48]
    assert info["image_size"] == [128, 384]
