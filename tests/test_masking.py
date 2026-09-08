import pytest
import torch

from inverse_psalm.data.masking import apply_masking, sample_schedule_value, schedule_value


def test_linear_schedule_value():
    spec = {"type": "linear", "start": 0.1, "end": 0.5}
    assert abs(schedule_value(spec, 0.5) - 0.3) < 1e-6


def test_random_mask_rate_one_masks_all_residues():
    input_ids = torch.tensor([[0, 5, 6, 2, 1]])
    residue_mask = torch.tensor([[0, 1, 1, 0, 0]])
    config = {
        "masking": {
            "schedule": {"type": "constant", "value": 1.0},
            "mask_type": "random",
        }
    }
    out = apply_masking(
        input_ids=input_ids,
        residue_mask=residue_mask,
        mask_token_id=32,
        config=config,
        progress=0.0,
        seed=1,
    )
    assert out["masked_input_ids"].tolist() == [[0, 32, 32, 2, 1]]
    assert out["mlm_labels"].tolist() == [[-100, 5, 6, -100, -100]]


def test_dsm_mixture_samples_within_configured_ranges():
    generator = torch.Generator()
    generator.manual_seed(1)
    spec = {
        "type": "dsm_mixture",
        "branches": [
            {"probability": 0.70, "mask_type": "random", "min_rate": 0.01, "max_rate": 0.99},
            {"probability": 0.15, "mask_type": "contiguous", "min_rate": 0.01, "max_rate": 0.15},
            {"probability": 0.15, "mask_type": "random", "min_rate": 0.75, "max_rate": 0.99},
        ],
    }
    values = [sample_schedule_value(spec, 1.0, generator, torch.device("cpu")) for _ in range(200)]
    assert all(0.01 <= value <= 0.99 for value in values)
    assert any(value <= 0.15 for value in values)
    assert any(value >= 0.75 for value in values)
    assert len(set(round(value, 4) for value in values)) > 1


def test_dsm_mixture_respects_single_branch_range():
    generator = torch.Generator()
    generator.manual_seed(1)
    spec = {
        "type": "dsm_mixture",
        "branches": [
            {"probability": 1.0, "mask_type": "random", "min_rate": 0.75, "max_rate": 0.99},
        ],
    }
    values = [sample_schedule_value(spec, 1.0, generator, torch.device("cpu")) for _ in range(20)]
    assert all(0.75 <= value <= 0.99 for value in values)


def test_dsm_mixture_rejects_invalid_ranges():
    generator = torch.Generator()
    spec = {
        "type": "dsm_mixture",
        "branches": [
            {"probability": 1.0, "mask_type": "random", "min_rate": 0.8, "max_rate": 0.2},
        ],
    }
    with pytest.raises(ValueError, match="max_rate"):
        sample_schedule_value(spec, 1.0, generator, torch.device("cpu"))


def test_random_mask_type_masks_requested_count():
    input_ids = torch.tensor([[0, 5, 6, 7, 8, 2]])
    residue_mask = torch.tensor([[0, 1, 1, 1, 1, 0]])
    config = {
        "masking": {
            "schedule": {"type": "constant", "value": 0.5},
            "mask_type": "random",
        }
    }
    out = apply_masking(
        input_ids=input_ids,
        residue_mask=residue_mask,
        mask_token_id=32,
        config=config,
        progress=0.0,
        seed=2,
    )
    assert out["mask_type"] == "random"
    assert int(out["mask_positions"].sum().item()) == 2


def test_dsm_mixture_branch_selects_mask_type():
    input_ids = torch.tensor([[0, 5, 6, 7, 8, 2]])
    residue_mask = torch.tensor([[0, 1, 1, 1, 1, 0]])
    config = {
        "masking": {
            "schedule": {
                "type": "dsm_mixture",
                "branches": [
                    {"probability": 1.0, "mask_type": "contiguous", "min_rate": 0.5, "max_rate": 0.5},
                ],
            },
        }
    }
    out = apply_masking(
        input_ids=input_ids,
        residue_mask=residue_mask,
        mask_token_id=32,
        config=config,
        progress=0.0,
        seed=2,
    )
    assert out["mask_type"] == "contiguous"
    selected = out["mask_positions"][0].nonzero(as_tuple=False).flatten()
    assert selected.numel() == 2
    assert torch.all(selected[1:] == selected[:-1] + 1)


def test_random_masking_never_masks_non_residues():
    input_ids = torch.tensor([[0, 5, 6, 7, 2]])
    residue_mask = torch.tensor([[0, 1, 1, 1, 0]])
    config = {
        "masking": {
            "schedule": {"type": "constant", "value": 1.0},
            "mask_type": "random",
        }
    }
    out = apply_masking(
        input_ids=input_ids,
        residue_mask=residue_mask,
        mask_token_id=32,
        config=config,
        progress=0.0,
        seed=5,
    )
    assert not bool(out["mask_positions"][0, 0])
    assert not bool(out["mask_positions"][0, -1])


def test_contiguous_span_masking_masks_one_block():
    input_ids = torch.tensor([[0, 5, 6, 7, 8, 9, 10, 2]])
    residue_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0]])
    config = {
        "masking": {
            "schedule": {"type": "constant", "value": 0.5},
            "mask_type": "contiguous",
        }
    }

    out = apply_masking(
        input_ids=input_ids,
        residue_mask=residue_mask,
        mask_token_id=32,
        config=config,
        progress=0.0,
        seed=3,
    )

    selected = out["mask_positions"][0].nonzero(as_tuple=False).flatten()
    assert selected.numel() == 3
    assert torch.all(selected[1:] == selected[:-1] + 1)
    assert out["masked_input_ids"][0, selected].tolist() == [32, 32, 32]


def test_contiguous_span_masking_clamps_to_longest_residue_run():
    input_ids = torch.tensor([[0, 5, 6, 7, 8, 9, 10, 2]])
    residue_mask = torch.tensor([[0, 1, 1, 1, 0, 1, 1, 0]])
    config = {
        "masking": {
            "schedule": {"type": "constant", "value": 1.0},
            "mask_type": "contiguous",
        }
    }

    out = apply_masking(
        input_ids=input_ids,
        residue_mask=residue_mask,
        mask_token_id=32,
        config=config,
        progress=0.0,
        seed=4,
    )

    selected = out["mask_positions"][0].nonzero(as_tuple=False).flatten()
    assert selected.numel() == 3
    assert torch.all(selected[1:] == selected[:-1] + 1)
    assert residue_mask[0, selected].all()
