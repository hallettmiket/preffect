"""Starting from a model someone already fitted.

Finding 36: version 0.1 had this, and its layer freezing could not freeze
anything. ``set_parameter_requires_grad`` reads

    if any(keyword in name for keyword in ["omega", "theta", "pi_layer"]):
        param.requires_grad = True
    else:
        param.requires_grad = True

Both branches are identical, so the condition cannot change the outcome. For
the single and full models the body says "IS NOT YET IMPLEMENTED" and also
sets everything trainable. A run that asked to freeze the encoder trained the
encoder, and looked exactly like one that had not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from preffect.checkpoint import save_checkpoint
from preffect.config import Likelihood, ModelConfig, ModelKind
from preffect.errors import CheckpointError
from preffect.model import SimpleModel
from preffect.pretraining import (
    freeze_parameters,
    load_pretrained,
    trainable_parameters,
)

N_GENES = 12


def build(**changes) -> SimpleModel:
    torch.manual_seed(0)
    return SimpleModel(N_GENES, ModelConfig(hidden_dim=16, latent_dim=4, **changes))


@pytest.fixture
def saved(tmp_path) -> Path:
    """A fitted model on disk, with weights that are not the initial ones."""
    model = build()
    with torch.no_grad():  # move the weights so "did it load" is answerable
        for parameter in model.parameters():
            parameter.add_(0.5)
    path = tmp_path / "pretrained.pt"
    save_checkpoint(path, model, model.config, N_GENES)
    return path


# --------------------------------------------------------------------------
# Weights are actually carried over.
# --------------------------------------------------------------------------


def test_the_weights_arrive(saved):
    fresh = build()
    before = fresh.encoder.to_mean.weight.clone()

    record = load_pretrained(fresh, saved)

    assert not torch.allclose(before, fresh.encoder.to_mean.weight)
    assert len(record.loaded) == len(list(fresh.named_parameters()))
    assert record.missing == () and record.unused == ()


def test_it_reports_what_it_did(saved):
    record = load_pretrained(build(), saved)
    assert "loaded" in str(record)
    assert record.source == saved
    assert record.written_by


def test_loading_twice_gives_the_same_model(saved):
    first, second = build(), build()
    load_pretrained(first, saved)
    load_pretrained(second, saved)
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(a, b)


# --------------------------------------------------------------------------
# Freezing freezes. This is the defect.
# --------------------------------------------------------------------------


def test_freezing_the_encoder_stops_it_training(saved):
    """The property version 0.1 claimed and did not have."""
    from preffect.config import (
        DataConfig,
        LossWeights,
        PreffectConfig,
        RuntimeConfig,
        TrainConfig,
    )
    from preffect.data.tensors import TensorData
    from preffect.training import train

    model = build()
    load_pretrained(model, saved, freeze=("encoder",))
    before = {n: p.clone() for n, p in model.named_parameters()}

    torch.manual_seed(0)
    counts = torch.randint(0, 50, (20, N_GENES)).float()
    train(
        model,
        TensorData(inputs=counts.clone(), targets=counts),
        PreffectConfig(
            data=DataConfig(train_path=Path("/nowhere")),
            model=model.config,
            loss=LossWeights(),
            train=TrainConfig(epochs=5, batch_size=10, learning_rate=0.05),
            runtime=RuntimeConfig(seed=0, num_threads=2),
        ),
    )

    moved = {
        name: not torch.allclose(before[name], parameter)
        for name, parameter in model.named_parameters()
    }
    assert not any(v for n, v in moved.items() if n.startswith("encoder")), (
        "the encoder was frozen and trained anyway"
    )
    assert any(v for n, v in moved.items() if not n.startswith("encoder")), (
        "nothing trained at all, so this proves nothing"
    )


def test_freezing_sets_requires_grad_off(saved):
    model = build()
    record = load_pretrained(model, saved, freeze=("encoder",))

    assert record.frozen, "nothing was recorded as frozen"
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (not name.startswith("encoder"))


def test_freezing_something_that_does_not_exist_is_refused(saved):
    """Version 0.1's freezing matched nothing and trained everything."""
    with pytest.raises(CheckpointError, match="no parameter starts with"):
        load_pretrained(build(), saved, freeze=("encodr",))


def test_the_refusal_lists_what_could_be_frozen(saved):
    with pytest.raises(CheckpointError) as raised:
        load_pretrained(build(), saved, freeze=("nonsense",))
    message = str(raised.value)
    assert "encoder" in message and "decoder" in message


def test_freezing_nothing_is_allowed_and_freezes_nothing(saved):
    model = build()
    record = load_pretrained(model, saved, freeze=())
    assert record.frozen == ()
    assert all(p.requires_grad for p in model.parameters())


def test_several_parts_can_be_frozen_at_once(saved):
    model = build()
    record = load_pretrained(model, saved, freeze=("encoder", "decoder"))
    assert any(n.startswith("encoder") for n in record.frozen)
    assert any(n.startswith("decoder") for n in record.frozen)


def test_the_optimiser_is_only_given_what_is_trainable(saved):
    model = build()
    load_pretrained(model, saved, freeze=("encoder",))
    assert len(trainable_parameters(model)) < len(list(model.parameters()))


def test_freezing_everything_is_refused_at_training_time(saved):
    """Otherwise the run looks fine and changes nothing."""
    from preffect.config import (
        DataConfig,
        LossWeights,
        PreffectConfig,
        RuntimeConfig,
        TrainConfig,
    )
    from preffect.data.tensors import TensorData
    from preffect.errors import ConfigError
    from preffect.training import train

    model = build()
    freeze_parameters(model, [n.split(".")[0] for n, _ in model.named_parameters()])

    counts = torch.randint(0, 50, (10, N_GENES)).float()
    with pytest.raises(ConfigError, match="every parameter .* is frozen"):
        train(
            model,
            TensorData(inputs=counts.clone(), targets=counts),
            PreffectConfig(
                data=DataConfig(train_path=Path("/nowhere")),
                model=model.config,
                loss=LossWeights(),
                train=TrainConfig(epochs=1, batch_size=5),
                runtime=RuntimeConfig(seed=0, num_threads=2),
            ),
        )


# --------------------------------------------------------------------------
# A mismatch says which field differs, rather than failing inside PyTorch.
# --------------------------------------------------------------------------


def test_a_different_gene_count_is_refused_by_name(saved):
    other = SimpleModel(N_GENES + 5, ModelConfig(hidden_dim=16, latent_dim=4))
    with pytest.raises(CheckpointError, match="fitted on 12 genes"):
        load_pretrained(other, saved)


def test_a_different_model_kind_is_refused(saved, tmp_path):
    model = build()
    # A checkpoint that records a different model kind.
    path = tmp_path / "single.pt"
    save_checkpoint(
        path, model, ModelConfig(hidden_dim=16, latent_dim=4, kind=ModelKind.SINGLE),
        N_GENES,
    )
    with pytest.raises(CheckpointError, match="'single' model and this one is"):
        load_pretrained(build(), path)


@pytest.mark.parametrize("setting,value", [("latent_dim", 8), ("hidden_dim", 32)])
def test_a_different_shape_is_refused_naming_the_setting(saved, setting, value):
    other = SimpleModel(N_GENES, ModelConfig(**{"hidden_dim": 16, "latent_dim": 4, setting: value}))
    with pytest.raises(CheckpointError, match=setting):
        load_pretrained(other, saved)


def test_missing_parameters_are_refused_unless_asked_for(tmp_path):
    """The usual cause is the wrong file, so it is not silently allowed."""
    plain = build(likelihood=Likelihood.NB)
    path = tmp_path / "plain.pt"
    save_checkpoint(path, plain, plain.config, N_GENES)

    zero_inflated = build(likelihood=Likelihood.ZINB)  # has an extra head
    with pytest.raises(CheckpointError, match="does not cover"):
        load_pretrained(zero_inflated, path)

    record = load_pretrained(zero_inflated, path, allow_missing=True)
    assert record.missing, "the extra head should be reported as missing"
    assert record.loaded, "the shared parts should still have loaded"
