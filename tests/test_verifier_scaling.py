import torch

from inverse_psalm.models.inverse_psalm import InversePSALMTrainingModule
from inverse_psalm.training.trainer import InversePSALMTrainer


def test_verifier_mask_scale_full_strength_then_clips():
    model = InversePSALMTrainingModule.__new__(InversePSALMTrainingModule)
    model.config = {
        "loss": {
            "verifier_mask_scale": {
                "enabled": True,
                "full_strength_until": 0.30,
                "min_scale": 0.10,
            }
        }
    }

    assert model._verifier_mask_scale(0.15, torch.device("cpu"), torch.float32).item() == 1.0
    assert model._verifier_mask_scale(0.30, torch.device("cpu"), torch.float32).item() == 1.0
    assert abs(model._verifier_mask_scale(0.65, torch.device("cpu"), torch.float32).item() - 0.5) < 1e-6
    assert abs(model._verifier_mask_scale(0.99, torch.device("cpu"), torch.float32).item() - 0.10) < 1e-6


def test_verifier_warmup_reaches_one_at_configured_step():
    trainer = InversePSALMTrainer.__new__(InversePSALMTrainer)
    trainer.config = {"loss": {"verifier_weight": {"warmup_steps": 5000}}}

    class State:
        global_step = 2500

    trainer.state = State()
    assert trainer._verifier_warmup() == 0.5
    trainer.state.global_step = 5000
    assert trainer._verifier_warmup() == 1.0
