import torch
import torch.nn as nn

from inverse_psalm.training.trainer import InversePSALMTrainer


def test_apply_example_class_loss_masks_respects_config():
    trainer = InversePSALMTrainer.__new__(InversePSALMTrainer)
    trainer.config = {
        "loss": {
            "example_classes": {
                "original": {"mlm_loss": True, "annotation_loss": "domain_only"},
                "negative": {"mlm_loss": False, "annotation_loss": "all_residues"},
            }
        }
    }
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "mlm_labels": torch.tensor([[-100, 2, -100], [-100, 5, -100]]),
        "annotation_ids": torch.tensor([[-100, 7, 0], [-100, 0, 0]]),
        "residue_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
        "domain_mask": torch.tensor([[0, 1, 0], [0, 0, 0]]),
        "example_types": ["original", "negative"],
    }

    out = trainer._apply_example_class_loss_masks(inputs)

    assert out["mlm_labels"].tolist() == [[-100, 2, -100], [-100, -100, -100]]
    assert out["conditioning_annotation_ids"].tolist() == [[-100, 7, -100], [-100, 0, 0]]
    assert out["annotation_loss_mask"].tolist() == [[False, True, False], [False, True, True]]


def test_apply_example_class_loss_masks_supports_domain_only_mlm():
    trainer = InversePSALMTrainer.__new__(InversePSALMTrainer)
    trainer.config = {
        "loss": {
            "example_classes": {
                "shuffled": {"mlm_loss": "domain_only", "annotation_loss": "all_residues"},
            }
        }
    }
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "mlm_labels": torch.tensor([[-100, 2, 3, 4]]),
        "annotation_ids": torch.tensor([[-100, 7, 8, 0]]),
        "residue_mask": torch.tensor([[0, 1, 1, 1]]),
        "domain_mask": torch.tensor([[0, 1, 0, 1]]),
        "example_types": ["shuffled"],
    }

    out = trainer._apply_example_class_loss_masks(inputs)

    assert out["mlm_labels"].tolist() == [[-100, 2, -100, 4]]
    assert out["conditioning_annotation_ids"].tolist() == [[-100, 7, 8, 0]]
    assert out["annotation_loss_mask"].tolist() == [[False, True, True, True]]


def test_global_annotation_loss_false_disables_annotation_loss_mask():
    trainer = InversePSALMTrainer.__new__(InversePSALMTrainer)
    trainer.config = {
        "model": {"annotation_loss": False},
        "loss": {
            "example_classes": {
                "shuffled": {"mlm_loss": "all_residues", "annotation_loss": "all_residues"},
            }
        },
    }
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "mlm_labels": torch.tensor([[-100, 2, 3, 4]]),
        "annotation_ids": torch.tensor([[-100, 7, 8, 0]]),
        "residue_mask": torch.tensor([[0, 1, 1, 1]]),
        "domain_mask": torch.tensor([[0, 1, 0, 1]]),
        "example_types": ["shuffled"],
    }

    out = trainer._apply_example_class_loss_masks(inputs)

    assert out["conditioning_annotation_ids"].tolist() == [[-100, 7, 8, 0]]
    assert out["annotation_loss_mask"].tolist() == [[False, False, False, False]]


def test_global_annotation_track_false_ignores_conditioning_ids():
    trainer = InversePSALMTrainer.__new__(InversePSALMTrainer)
    trainer.config = {
        "model": {"annotation_track": False, "annotation_loss": False},
        "loss": {
            "example_classes": {
                "shuffled": {"mlm_loss": "all_residues", "annotation_loss": "all_residues"},
            }
        },
    }
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "mlm_labels": torch.tensor([[-100, 2, 3, 4]]),
        "annotation_ids": torch.tensor([[-100, 7, 8, 0]]),
        "residue_mask": torch.tensor([[0, 1, 1, 1]]),
        "domain_mask": torch.tensor([[0, 1, 0, 1]]),
        "example_types": ["shuffled"],
    }

    out = trainer._apply_example_class_loss_masks(inputs)

    assert out["conditioning_annotation_ids"].tolist() == [[-100, -100, -100, -100]]
    assert out["annotation_loss_mask"].tolist() == [[False, False, False, False]]


def test_original_conditioning_supports_fully_unconditional_branch():
    trainer = InversePSALMTrainer.__new__(InversePSALMTrainer)
    trainer.config = {
        "loss": {
            "example_classes": {
                "original": {
                    "mlm_loss": "all_residues",
                    "annotation_loss": "domain_only",
                    "conditioning": "domain_only_or_unconditional",
                    "unconditional_prob": 1.0,
                },
            }
        }
    }
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "mlm_labels": torch.tensor([[-100, 2, 3, 4]]),
        "annotation_ids": torch.tensor([[-100, 7, 0, 8]]),
        "residue_mask": torch.tensor([[0, 1, 1, 1]]),
        "domain_mask": torch.tensor([[0, 1, 0, 1]]),
        "example_types": ["original"],
    }

    out = trainer._apply_example_class_loss_masks(inputs, conditioning_seed=123)

    assert out["conditioning_annotation_ids"].tolist() == [[-100, -100, -100, -100]]
    assert out["annotation_loss_mask"].tolist() == [[False, False, False, False]]
    assert out["annotation_ids"].tolist() == [[-100, 7, 0, 8]]


def test_eval_override_forces_original_domain_only_conditioning():
    trainer = InversePSALMTrainer.__new__(InversePSALMTrainer)
    trainer.config = {
        "loss": {
            "example_classes": {
                "original": {
                    "mlm_loss": "all_residues",
                    "annotation_loss": "domain_only",
                    "conditioning": "domain_only_or_unconditional",
                    "unconditional_prob": 1.0,
                },
            }
        }
    }
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "mlm_labels": torch.tensor([[-100, 2, 3, 4]]),
        "annotation_ids": torch.tensor([[-100, 7, 0, 8]]),
        "residue_mask": torch.tensor([[0, 1, 1, 1]]),
        "domain_mask": torch.tensor([[0, 1, 0, 1]]),
        "example_types": ["original"],
    }

    out = trainer._apply_example_class_loss_masks(
        inputs,
        conditioning_seed=123,
        original_conditioning_override="domain_only",
    )

    assert out["conditioning_annotation_ids"].tolist() == [[-100, 7, -100, 8]]
    assert out["annotation_loss_mask"].tolist() == [[False, True, False, True]]


def test_negative_conditioning_preserves_explicit_none():
    trainer = InversePSALMTrainer.__new__(InversePSALMTrainer)
    trainer.config = {
        "loss": {
            "example_classes": {
                "negative": {
                    "mlm_loss": "none",
                    "annotation_loss": "all_residues",
                    "conditioning": "all_residues",
                },
            }
        }
    }
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "mlm_labels": torch.tensor([[-100, 2, 3, 4]]),
        "annotation_ids": torch.tensor([[-100, 0, 0, 0]]),
        "residue_mask": torch.tensor([[0, 1, 1, 1]]),
        "domain_mask": torch.tensor([[0, 0, 0, 0]]),
        "example_types": ["negative"],
    }

    out = trainer._apply_example_class_loss_masks(inputs)

    assert out["mlm_labels"].tolist() == [[-100, -100, -100, -100]]
    assert out["conditioning_annotation_ids"].tolist() == [[-100, 0, 0, 0]]


def test_create_optimizer_uses_separate_annotation_gate_lr_when_configured():
    trainer = InversePSALMTrainer.__new__(InversePSALMTrainer)
    trainer.optimizer = None
    trainer.model = nn.Module()
    trainer.model.generator = nn.Module()
    trainer.model.generator.annotation = nn.Linear(1, 1)
    trainer.model.generator.annotation_gate_logits = nn.ParameterDict(
        {"block_01": nn.Parameter(torch.tensor(0.0))}
    )
    trainer.config = {
        "training": {
            "learning_rate": {
                "annotation": 2e-4,
                "annotation_gate": 1e-3,
            },
            "optimizer": {
                "weight_decay": {
                    "annotation": 0.0,
                    "annotation_gate": 0.0,
                    "default": 0.01,
                }
            },
        }
    }

    optimizer = trainer.create_optimizer()

    lrs = sorted(group["lr"] for group in optimizer.param_groups)
    assert lrs == [2e-4, 1e-3]
