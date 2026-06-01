import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDAD = os.path.join(ROOT, "indad")
if INDAD not in sys.path:
    sys.path.insert(0, INDAD)

from models import load_patchcore_archive, save_patchcore_archive, save_tensor


def test_load_patchcore_archive_reads_old_single_parameter_archive(tmp_path):
    patch_lib = torch.randn(2, 3, 4, 5)
    save_tensor(str(tmp_path), "old.ts", patch_lib)

    loaded_patch_lib, stats = load_patchcore_archive(str(tmp_path / "old.ts"))

    assert torch.allclose(loaded_patch_lib, patch_lib)
    assert stats is None


def test_load_patchcore_archive_reads_new_archive_with_embedded_stats(tmp_path):
    patch_lib = torch.randn(2, 3, 4, 5)
    stats = {
        "baseline": torch.ones(2, 3),
        "scale": torch.full((2, 3), 2.0),
        "recommended_pixel_threshold": torch.tensor(3.5),
    }
    save_patchcore_archive(str(tmp_path), "new.ts", patch_lib, stats)

    loaded_patch_lib, loaded_stats = load_patchcore_archive(str(tmp_path / "new.ts"))

    assert torch.allclose(loaded_patch_lib, patch_lib)
    assert torch.allclose(loaded_stats["baseline"], stats["baseline"])
    assert torch.allclose(loaded_stats["scale"], stats["scale"])
    assert torch.allclose(
        loaded_stats["recommended_pixel_threshold"],
        stats["recommended_pixel_threshold"],
    )
