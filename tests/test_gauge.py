import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "indad"))
from gauge import (_orthogonal_transport, fit_gauge_stats, loop_curvature,
                   loops_to_nodes, normalize_gauge)


def test_constant_features_have_zero_curvature_and_correct_shapes():
    features = torch.ones(2, 8, 4, 5)
    loops = loop_curvature(features, rank=2, chunk_size=5)
    assert loops.shape == (2, 3, 4)
    assert torch.isfinite(loops).all()
    assert torch.allclose(loops, torch.zeros_like(loops), atol=1e-4)
    assert loops_to_nodes(loops).shape == (2, 4, 5)


def test_loop_distribution_averages_incident_loops():
    loops = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])
    nodes = loops_to_nodes(loops)
    assert nodes[0, 1, 1] == 4.0
    assert nodes[0, 0, 1] == 2.0
    assert nodes[0, 2, 2] == 7.0


def test_transport_recovers_basis_change():
    frame = torch.eye(5)[:, :2].unsqueeze(0)
    rotation = torch.tensor([[[0.0, -1.0], [1.0, 0.0]]])
    assert torch.allclose(_orthogonal_transport(frame, frame @ rotation), rotation)


def test_invalid_lattice_is_rejected():
    with pytest.raises(ValueError, match="2x2"):
        loop_curvature(torch.zeros(1, 8, 1, 3))


@pytest.mark.parametrize("mode", ["global", "same_row", "exact_position"])
def test_calibration_matches_mode_and_rejects_wrong_mode(mode):
    normal = torch.rand(6, 3, 4)
    stats = fit_gauge_stats(normal, mode)
    output = normalize_gauge(normal[:2], stats, mode)
    assert output.shape == (2, 3, 4)
    assert torch.isfinite(output).all()
    with pytest.raises(ValueError, match="different match mode"):
        normalize_gauge(normal[:1], stats, "global" if mode != "global" else "same_row")
