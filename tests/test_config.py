"""Configuration tests.

Most of these exist because of a specific defect in the previous version. Where
that is so, the test name says which, and `how_this_project_breaks.md` has the
full account. A test here failing means a defect we have already paid for once
has come back.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from preffect import (
    ConfigError,
    CorrectionVariable,
    DataConfig,
    Dispersion,
    LossWeights,
    Masking,
    ModelConfig,
    ModelKind,
    PreffectConfig,
    RuntimeConfig,
    TrainConfig,
)


def make_config(**changes):
    """A minimal valid configuration, with sections overridden by keyword."""
    base = {
        "data": DataConfig(train_path=Path("/nowhere/train")),
        "model": ModelConfig(),
        "loss": LossWeights(),
        "train": TrainConfig(),
        "runtime": RuntimeConfig(),
    }
    base.update(changes)
    return PreffectConfig(**base)


# --------------------------------------------------------------------------
# The model you ask for is the model you get.
#
# Defect: asking for the single-tissue graph model produced the plain model
# instead, silently, because the check ran before the data was read and so
# could not see that an adjacency network was present. The adjacency data was
# then loaded and ignored.
# --------------------------------------------------------------------------


def test_full_model_requires_more_than_one_tissue():
    with pytest.raises(ConfigError, match="at least 2"):
        make_config(
            data=DataConfig(train_path=Path("/nowhere/train"), n_tissues=1),
            model=ModelConfig(kind=ModelKind.FULL),
        )


@pytest.mark.parametrize("kind", [ModelKind.SIMPLE, ModelKind.SINGLE])
def test_single_tissue_models_refuse_several_tissues(kind):
    with pytest.raises(ConfigError, match="exactly one tissue"):
        make_config(
            data=DataConfig(train_path=Path("/nowhere/train"), n_tissues=3),
            model=ModelConfig(kind=kind),
            loss=LossWeights().for_tissue_count(3),
        )


def test_requested_kind_survives_construction():
    cfg = make_config(model=ModelConfig(kind=ModelKind.SINGLE))
    assert cfg.model.kind is ModelKind.SINGLE
    assert cfg.model.kind.uses_adjacency


# --------------------------------------------------------------------------
# Per-tissue weights are checked against the number of tissues.
#
# Defect: the weight lists were built once at import with one tissue, nothing
# rebuilt them, and the weighting step paired them against the per-tissue
# losses with zip, which stops at the shorter. A two-tissue run with a
# one-entry list dropped the second tissue out of the objective entirely.
# --------------------------------------------------------------------------


def test_short_weight_list_is_refused_not_truncated():
    with pytest.raises(ConfigError, match="one entry per tissue"):
        PreffectConfig(
            data=DataConfig(train_path=Path("/nowhere/train"), n_tissues=2),
            model=ModelConfig(kind=ModelKind.FULL),
            loss=LossWeights(adjacency_recon=(100.0,)),  # one value, two tissues
        )


def test_error_names_every_offending_weight_and_both_counts():
    with pytest.raises(ConfigError) as excinfo:
        LossWeights().check_tissue_count(3)
    message = str(excinfo.value)
    assert "3 tissues" in message
    for name in LossWeights.PER_TISSUE:
        assert name in message


def test_for_tissue_count_expands_a_single_value():
    weights = LossWeights().for_tissue_count(3)
    assert weights.adjacency_recon == (100.0, 100.0, 100.0)
    assert weights.expression_kl == (0.1, 0.1, 0.1)
    weights.check_tissue_count(3)


def test_for_tissue_count_refuses_an_ambiguous_length():
    with pytest.raises(ConfigError, match="cannot be expanded"):
        LossWeights(adjacency_recon=(1.0, 2.0)).for_tissue_count(3)


def test_for_tissue_count_leaves_correct_lengths_alone():
    weights = LossWeights(adjacency_recon=(1.0, 2.0)).for_tissue_count(2)
    assert weights.adjacency_recon == (1.0, 2.0)


def test_weights_must_be_positive():
    with pytest.raises(ConfigError, match="positive"):
        LossWeights(adjacency_recon=(0.0,))
    with pytest.raises(ConfigError, match="positive"):
        LossWeights(expression_recon=-1.0)


def test_a_bare_number_is_accepted_for_a_single_tissue_weight():
    weights = LossWeights(library_recon=5)
    assert weights.library_recon == (5.0,)


# --------------------------------------------------------------------------
# Validation is optional.
#
# Defect: a validation folder had to exist before the code would run, so a
# cohort too small to split was handled by putting the same file in both
# folders, and the resulting validation loss was the training loss.
# --------------------------------------------------------------------------


def test_validation_path_may_be_omitted():
    data = DataConfig(train_path=Path("/nowhere/train"))
    assert data.validation_path is None
    assert not data.has_validation


def test_validation_path_is_reported_when_given():
    data = DataConfig(
        train_path=Path("/nowhere/train"), validation_path=Path("/nowhere/validation")
    )
    assert data.has_validation


def test_paths_have_no_defaults():
    with pytest.raises(TypeError):
        DataConfig()  # train_path is required


def test_string_paths_become_path_objects():
    data = DataConfig(train_path="/nowhere/train", validation_path="/nowhere/validation")
    assert isinstance(data.train_path, Path)
    assert isinstance(data.validation_path, Path)


# --------------------------------------------------------------------------
# Every sample is used unless you say otherwise.
#
# Defect: the final incomplete batch of each epoch was discarded without a
# word, so with 250 samples and a batch of 200, fifty never entered training.
# --------------------------------------------------------------------------


def test_the_last_partial_batch_is_kept_by_default():
    assert TrainConfig().drop_last_batch is False


# --------------------------------------------------------------------------
# One device, resolved once.
#
# Defect: the device was rebuilt from the configuration in six places, and two
# of them ignored the request for a CPU, so a CPU run on a machine with a GPU
# moved some tensors to the GPU and then failed on a device mismatch.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "cuda:7"])
def test_accepted_devices(device):
    assert RuntimeConfig(device=device).device == device


@pytest.mark.parametrize("device", ["gpu", "cuda0", "mps", ""])
def test_rejected_devices(device):
    with pytest.raises(ConfigError, match="device must be"):
        RuntimeConfig(device=device)


def test_uses_cuda_reflects_the_device():
    assert RuntimeConfig(device="cuda:3").uses_cuda
    assert not RuntimeConfig(device="cpu").uses_cuda


def test_determinism_without_a_seed_is_refused():
    with pytest.raises(ConfigError, match="seed is None"):
        RuntimeConfig(deterministic=True)
    RuntimeConfig(deterministic=True, seed=42)  # fine


# --------------------------------------------------------------------------
# Settings that contradict each other are refused, not reconciled.
# --------------------------------------------------------------------------


def test_hidden_dim_must_exceed_latent_dim():
    with pytest.raises(ConfigError, match="must be larger than"):
        ModelConfig(latent_dim=50, hidden_dim=50)


def test_gene_batch_dispersion_needs_a_batch_variable():
    with pytest.raises(ConfigError, match="gene-batch"):
        ModelConfig(dispersion=Dispersion.GENE_BATCH)

    ModelConfig(
        dispersion=Dispersion.GENE_BATCH,
        correction_variables=(CorrectionVariable("batch", "categorical"),),
    )


def test_correction_variable_kind_is_checked():
    with pytest.raises(ConfigError, match="categorical.*continuous"):
        CorrectionVariable("batch", "catagorical")  # codespell:ignore


def test_duplicate_correction_variables_are_refused():
    with pytest.raises(ConfigError, match="more than once"):
        ModelConfig(
            correction_variables=(
                CorrectionVariable("batch", "categorical"),
                CorrectionVariable("batch", "continuous"),
            )
        )


def test_masking_and_fraction_must_agree():
    with pytest.raises(ConfigError, match="masking is 'none'"):
        DataConfig(train_path=Path("/x"), mask_fraction=0.1)
    with pytest.raises(ConfigError, match="nothing"):
        DataConfig(train_path=Path("/x"), masking=Masking.MCAR, mask_fraction=0.0)

    DataConfig(train_path=Path("/x"), masking=Masking.MCAR, mask_fraction=0.1)


def test_mask_fraction_is_a_proportion():
    with pytest.raises(ConfigError, match=r"\[0, 1\]"):
        DataConfig(train_path=Path("/x"), masking=Masking.MCAR, mask_fraction=1.5)


# --------------------------------------------------------------------------
# A configuration cannot be edited after it is built.
#
# Defect: configuration was a module-level dictionary that the data loader,
# the batch iterator and the session factory all modified in place, so two
# models built in one session shared state.
# --------------------------------------------------------------------------


def test_sections_are_frozen():
    cfg = make_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.model.latent_dim = 999
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.train.epochs = 1


def test_replace_returns_a_new_object_and_rechecks():
    cfg = make_config()
    other = cfg.replace(train=TrainConfig(epochs=5))
    assert cfg.train.epochs != 5
    assert other.train.epochs == 5

    with pytest.raises(ConfigError, match="at least 2"):
        cfg.replace(model=ModelConfig(kind=ModelKind.FULL))


def test_replace_refuses_an_unknown_section():
    with pytest.raises(ConfigError, match="unknown configuration section"):
        make_config().replace(modle=ModelConfig())  # codespell:ignore


# --------------------------------------------------------------------------
# Defaults match what the paper reports, not what the old release shipped.
# --------------------------------------------------------------------------


def test_library_size_inference_is_off_by_default():
    """Supplementary Table S3 gives this default as Off.

    The published release shipped it as True, which is the switch that exposed
    the dropped library reconstruction term in the graph models.
    """
    assert ModelConfig().infer_library_size is False


# --------------------------------------------------------------------------
# What early stopping watches.
#
# Finding 50: the loss early stopping watches bottoms out about ten times
# sooner than imputation of the fitted cohort settles, so a rule that watches
# the imputation error on the hidden entries exists beside it (issue #28). A
# misspelt choice, or the imputation rule with no patience to act on it, is
# refused rather than ignored.
# --------------------------------------------------------------------------


def test_stop_on_is_loss_or_imputation():
    with pytest.raises(ConfigError, match="stop_on"):
        TrainConfig(stop_on="latent")


def test_stop_on_imputation_needs_a_patience():
    with pytest.raises(ConfigError, match="early_stopping_patience"):
        TrainConfig(stop_on="imputation")
    assert TrainConfig(stop_on="imputation", early_stopping_patience=5).uses_early_stopping



# --------------------------------------------------------------------------
# The network term's balance is a named setting with three values.
#
# Issue #13: the term is almost entirely "no edge here" on a real network, and
# version 0.1 carried weights for present and absent edges that nothing set.
# Here the choice is one setting, validated, and any other value is refused by
# name rather than read as "none".
# --------------------------------------------------------------------------


def test_edge_balance_defaults_to_balanced():
    """The decision of 2026-10-30: under 'none' the network half learned nothing."""
    assert LossWeights().edge_balance == "balanced"


@pytest.mark.parametrize("mode", ["none", "balanced"])
def test_edge_balance_accepts_its_two_modes(mode):
    assert LossWeights(edge_balance=mode).edge_balance == mode


@pytest.mark.parametrize("mode", ["weighted", "sampled", "NONE", "", "balance"])
def test_edge_balance_refuses_anything_else_by_name(mode):
    """'sampled' is among the refused: it lost exp/17's comparison and lives
    in that experiment, not in the package, by the rule in exp/README.md."""
    with pytest.raises(ConfigError, match="edge_balance must be one of") as excinfo:
        LossWeights(edge_balance=mode)
    assert repr(mode) in str(excinfo.value)


def test_edge_balance_survives_the_tissue_count_expansion():
    assert LossWeights(edge_balance="balanced").for_tissue_count(3).edge_balance == "balanced"
