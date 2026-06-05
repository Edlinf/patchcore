import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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


from patchcore_predict_simple import (
    apply_score_stats,
    parse_model_info_simple,
    raw_map_exact_position,
    raw_map_global,
    raw_map_same_row,
    select_score_map,
    split_big_image_by_geometry,
    stitch_tile_score,
)


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
    assert info["image_size"] == [384, 128]


def test_raw_map_global_matches_any_position():
    patch = torch.tensor([[[[0.0, 10.0]]]])
    patch_lib = torch.tensor([[[[5.0]], [[10.0]]]])

    raw = raw_map_global(patch, patch_lib)

    assert torch.allclose(raw, torch.tensor([[5.0, 0.0]]))


def test_raw_map_exact_position_radius_zero_matches_same_position():
    patch = torch.tensor([[[[0.0, 10.0]]]])
    patch_lib = torch.tensor([[[[5.0]], [[7.0]]]])

    raw = raw_map_exact_position(patch, patch_lib, neighbor_radius=0)

    assert torch.allclose(raw, torch.tensor([[5.0, 3.0]]))


def test_raw_map_exact_position_radius_one_can_match_neighbor_position():
    patch = torch.tensor([[[[0.0, 10.0]]]])
    patch_lib = torch.tensor([[[[5.0]], [[10.0]]]])

    raw = raw_map_exact_position(patch, patch_lib, neighbor_radius=1)

    assert torch.allclose(raw, torch.tensor([[5.0, 0.0]]))


def test_raw_map_same_row_radius_zero_matches_same_row():
    patch = torch.tensor([[[[0.0, 10.0], [20.0, 30.0]]]])
    patch_lib = torch.tensor([
        [[[1.0]], [[11.0]]],
        [[[100.0]], [[31.0]]],
    ])

    raw = raw_map_same_row(patch, patch_lib, neighbor_radius=0)

    assert torch.allclose(raw, torch.tensor([[1.0, 1.0], [11.0, 1.0]]))


def test_raw_map_same_row_radius_one_can_match_neighbor_row():
    patch = torch.tensor([[[[0.0], [20.0]]]])
    patch_lib = torch.tensor([
        [[[100.0]]],
        [[[1.0]]],
    ])

    raw = raw_map_same_row(patch, patch_lib, neighbor_radius=1)

    assert torch.allclose(raw, torch.tensor([[1.0], [19.0]]))


def test_select_score_map_applies_stats_only_for_exact_position():
    raw = torch.tensor([[3.0]])
    stats = {"baseline": torch.tensor([[1.0]]), "scale": torch.tensor([[2.0]])}

    exact = select_score_map(raw, stats, match_mode="exact_position")
    global_map = select_score_map(raw, stats, match_mode="global")

    assert torch.allclose(exact, torch.tensor([[1.0]]))
    assert torch.allclose(global_map, raw)


def test_split_big_image_by_geometry_matches_prepare_py_layout():
    arr = np.zeros((8, 10, 3), dtype=np.uint8)
    arr[2:4, 1:4] = [10, 0, 0]
    arr[2:4, 5:8] = [20, 0, 0]
    arr[5:7, 1:4] = [30, 0, 0]
    arr[5:7, 5:8] = [40, 0, 0]
    image = Image.fromarray(arr, "RGB")

    tiles = split_big_image_by_geometry(
        image,
        rows=2,
        cols=2,
        top_margin=2,
        bottom_margin=1,
        left_margin=1,
        right_margin=2,
        hori_gap=1,
        vert_gap=1,
    )

    assert len(tiles) == 4
    assert tiles[0].box == (1, 2, 4, 4)
    assert tiles[1].box == (5, 2, 8, 4)
    assert tiles[2].box == (1, 5, 4, 7)
    assert tiles[3].box == (5, 5, 8, 7)
    assert np.asarray(tiles[3].image)[0, 0].tolist() == [40, 0, 0]


def test_stitch_tile_score_keeps_margins_and_gaps_zero():
    full = np.zeros((8, 10), dtype=np.float32)
    tile_score = torch.ones(2, 3)

    stitch_tile_score(full, tile_score, box=(1, 2, 4, 4))

    assert full[0].sum() == 0
    assert full[:, 0].sum() == 0
    assert np.allclose(full[2:4, 1:4], 1.0)
    assert full[4].sum() == 0
