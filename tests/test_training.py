"""The training loop, and the number it exists to get right."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from preffect.config import (
    DataConfig,
    LossWeights,
    ModelConfig,
    PreffectConfig,
    RuntimeConfig,
    TrainConfig,
)
from preffect.data.tensors import TensorData
from preffect.errors import ConfigError
from preffect.model import SimpleModel
from preffect.training import run_epoch, train

N_GENES = 20


def data(n_samples: int = 30, seed: int = 0) -> TensorData:
    rng = np.random.default_rng(seed)
    counts = torch.from_numpy(
        rng.integers(0, 60, size=(n_samples, N_GENES)).astype(np.float32)
    )
    return TensorData(inputs=counts.clone(), targets=counts)


def config(**train_changes) -> PreffectConfig:
    return PreffectConfig(
        data=DataConfig(train_path=Path("/nowhere")),
        model=ModelConfig(hidden_dim=16, latent_dim=6),
        loss=LossWeights(),
        train=TrainConfig(epochs=3, batch_size=10, **train_changes),
        runtime=RuntimeConfig(seed=0),
    )


def model() -> SimpleModel:
    torch.manual_seed(0)
    return SimpleModel(N_GENES, ModelConfig(hidden_dim=16, latent_dim=6))


# --------------------------------------------------------------------------
# An epoch's loss is the mean over its batches.
#
# Defect: version 0.1 divided the accumulated validation loss by the number of
# tissues, one for this model, not by the number of batches. The figure it
# printed, plotted and compared against for early stopping was a sum presented
# as a mean, and so was not comparable across batch sizes.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_samples", "batch_size", "expected_batches"),
    [(30, 10, 3), (123, 50, 3), (100, 100, 1), (7, 3, 3)],
)
def test_the_epoch_loss_is_the_mean_over_the_batches_it_ran(
    n_samples, batch_size, expected_batches
):
    cfg = config()
    cfg = cfg.replace(train=TrainConfig(epochs=1, batch_size=batch_size))
    split = data(n_samples)
    m = model()
    m.eval()

    # The model draws a sample in its reparameterisation step, so two passes
    # over the same data differ. Both passes below are seeded identically, and
    # neither shuffles, so the comparison is of arithmetic rather than of luck.
    torch.manual_seed(99)
    result = run_epoch(m, split, cfg, epoch=1)
    assert result.n_batches == expected_batches

    # Recompute the mean independently from the per-batch losses.
    from preffect.data.batching import batch_indices
    from preffect.loss import compute_loss

    totals = []
    torch.manual_seed(99)
    with torch.no_grad():
        for positions in batch_indices(
            n_samples, batch_size, shuffle=False, drop_last=False
        ):
            part = split.select(positions)
            totals.append(
                float(
                    compute_loss(
                        m(part.inputs, part.corrections), part.targets, cfg.loss
                    ).total
                )
            )

    assert result.mean_loss == pytest.approx(sum(totals) / len(totals), rel=1e-5)


def test_the_batch_count_is_recorded_so_the_arithmetic_can_be_checked():
    cfg = config()
    history = train(model(), data(30), cfg)
    for record in history.epochs:
        assert record.train.n_batches == 3
        assert record.train.n_samples == 30


def test_a_different_batch_size_does_not_rescale_the_reported_loss():
    """The heart of the defect: version 0.1's figure scaled with batch count."""
    split = data(60)
    results = {}
    for batch_size in (10, 20, 30):
        cfg = config().replace(train=TrainConfig(epochs=1, batch_size=batch_size))
        m = model()
        m.eval()
        results[batch_size] = run_epoch(m, split, cfg, epoch=1).mean_loss

    values = list(results.values())
    assert max(values) / min(values) < 1.10, results


def test_both_splits_go_through_the_same_function():
    """Version 0.1 wrote the two passes out separately and they drifted."""
    cfg = config()
    history = train(model(), data(30), cfg, validation_data=data(20, seed=1))
    for record in history.epochs:
        assert record.validation is not None
        assert record.validation.n_batches == 2
        assert record.train.n_batches == 3


# --------------------------------------------------------------------------
# Training does what it says.
# --------------------------------------------------------------------------


def test_the_loss_falls():
    cfg = config().replace(train=TrainConfig(epochs=25, batch_size=10, learning_rate=1e-2))
    history = train(model(), data(30), cfg)
    assert history.train_losses[-1] < history.train_losses[0]


def test_every_epoch_is_recorded():
    history = train(model(), data(30), config())
    assert len(history) == 3
    assert [e.epoch for e in history.epochs] == [1, 2, 3]


def test_each_loss_term_is_reported_per_epoch():
    history = train(model(), data(30), config())
    terms = history.epochs[0].train.mean_terms
    assert set(terms) == {"expression_recon", "expression_kl"}


def test_the_callback_sees_every_epoch():
    seen = []
    train(model(), data(30), config(), on_epoch=seen.append)
    assert [r.epoch for r in seen] == [1, 2, 3]


def test_a_seed_makes_the_run_repeatable():
    a = train(model(), data(30), config())
    b = train(model(), data(30), config())
    assert a.train_losses == b.train_losses


# --------------------------------------------------------------------------
# Early stopping.
# --------------------------------------------------------------------------


def test_early_stopping_is_off_unless_asked_for():
    history = train(model(), data(30), config().replace(
        train=TrainConfig(epochs=4, batch_size=10)
    ))
    assert not history.stopped_early
    assert len(history) == 4


def test_early_stopping_halts_and_records_why():
    cfg = config().replace(
        train=TrainConfig(
            epochs=50,
            batch_size=10,
            learning_rate=1e-9,          # too small to improve anything
            early_stopping_patience=2,
        )
    )
    history = train(model(), data(30), cfg, validation_data=data(20, seed=1))
    assert history.stopped_early
    assert len(history) < 50
    assert history.best_epoch is not None


def test_early_stopping_on_the_training_loss_is_flagged():
    """Stopping on the training loss is a weaker thing and should not pass unremarked."""
    cfg = config().replace(
        train=TrainConfig(epochs=5, batch_size=10, early_stopping_patience=2)
    )
    history = train(model(), data(30), cfg, validation_data=None)
    assert history.watched_training_loss


def test_without_early_stopping_nothing_is_flagged():
    history = train(model(), data(30), config())
    assert not history.watched_training_loss


# --------------------------------------------------------------------------
# Refusals.
# --------------------------------------------------------------------------


def test_a_split_that_produces_no_batches_is_refused():
    cfg = config()
    empty = TensorData(inputs=torch.zeros(0, N_GENES), targets=torch.zeros(0, N_GENES))
    with pytest.raises(ConfigError, match="produced no batches"):
        run_epoch(model(), empty, cfg, epoch=1)


def test_mismatched_tensor_shapes_are_refused():
    from preffect.errors import DataError

    with pytest.raises(DataError, match="have to match"):
        TensorData(inputs=torch.zeros(5, 10), targets=torch.zeros(5, 11))


def test_correction_rows_must_match_the_samples():
    from preffect.errors import DataError

    with pytest.raises(DataError, match="3 rows but there are 5 samples"):
        TensorData(
            inputs=torch.zeros(5, 10),
            targets=torch.zeros(5, 10),
            corrections=torch.zeros(3, 1),
        )


# --------------------------------------------------------------------------
# train() must set up the run itself, not rely on its caller having done it.
#
# Finding 31: deleting configure_runtime from train passed the whole suite.
# test_a_seed_makes_the_run_repeatable did not notice, because its model()
# helper calls torch.manual_seed(0) before building, so both runs start from
# identical weights whatever train does. The command line kept working because
# cli.command_train makes its own call. Nothing covered the library path, which
# is the entire remedy for finding 24 and half of finding 26's.
# --------------------------------------------------------------------------


def test_train_seeds_the_run_itself_without_help_from_the_caller():
    """Identical starting weights, no seeding outside train, same result twice.

    The weights are copied rather than re-seeded, so the only thing that can
    make the two runs differ is the shuffling and the latent noise draws, which
    is exactly what train is responsible for seeding.
    """
    import copy

    torch.manual_seed(0)
    original = model()
    cfg = config().replace(runtime=RuntimeConfig(seed=11, num_threads=2))

    first = train(copy.deepcopy(original), data(30), cfg)
    torch.manual_seed(123)  # deliberately disturb the global state in between
    second = train(copy.deepcopy(original), data(30), cfg)

    assert first.train_losses == second.train_losses, (
        "train did not seed the run; it is relying on the caller to have done it"
    )


def test_train_applies_the_thread_setting():
    """The other half of configure_runtime, and the one finding 24 turns on."""
    import torch as torch_module

    before = torch_module.get_num_threads()
    try:
        # Built at the suite's count and trained at another on purpose: that
        # is the thread change this test exists to see applied. Since issue
        # #23 that order also warns, and the warning is expected here.
        with pytest.warns(RuntimeWarning, match="built while PyTorch was using"):
            train(model(), data(30), config().replace(
                runtime=RuntimeConfig(seed=0, num_threads=3)
            ))
        assert torch_module.get_num_threads() == 3
    finally:
        torch_module.set_num_threads(before)


# --------------------------------------------------------------------------
# A collapsed latent space is reported while it happens. Issue #20, finding 41.
#
# The latent collapsing means no dimension varies between samples, so it
# carries nothing about which sample is which and everything downstream is
# describing noise. The loss does not show it: the expression terms keep
# falling. It was found here by measuring after the fact, which is the hard way.
# --------------------------------------------------------------------------


def test_every_epoch_records_how_many_latent_dimensions_are_alive():
    history = train(model(), data(30), config())
    for record in history.epochs:
        assert record.train.latent_dim > 0
        assert 0 <= record.train.active_latent_units <= record.train.latent_dim
        assert record.train.latent_rms >= 0


def test_a_healthy_run_is_not_reported_as_collapsed():
    history = train(model(), data(30), config())
    assert not history.latent_collapsed
    assert history.first_collapsed_epoch is None
    assert all(not r.train.latent_collapsed for r in history.epochs)


def test_a_collapsed_latent_is_detected_and_warned_about():
    """Forced by holding the latent mean at a constant, which is what a
    collapse looks like from the outside: no variation between samples."""
    import warnings as warnings_module

    import torch as torch_module

    m = model()
    # Every sample gets the same latent mean, so no dimension varies.
    m.encoder.to_mean = torch_module.nn.Linear(m.encoder.to_mean.in_features,
                                               m.encoder.to_mean.out_features)
    with torch_module.no_grad():
        m.encoder.to_mean.weight.zero_()
        m.encoder.to_mean.bias.zero_()
    m.encoder.to_mean.weight.requires_grad_(False)
    m.encoder.to_mean.bias.requires_grad_(False)

    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        history = train(m, data(30), config())

    assert history.latent_collapsed, "the collapse was not detected"
    assert history.first_collapsed_epoch == 1
    assert any("latent space collapsed" in str(w.message) for w in caught), (
        "it detected the collapse and said nothing"
    )


def test_the_collapse_warning_says_what_to_try():
    """A warning with no remedy makes the reader search rather than act."""
    import warnings as warnings_module

    import torch as torch_module

    m = model()
    with torch_module.no_grad():
        m.encoder.to_mean.weight.zero_()
        m.encoder.to_mean.bias.zero_()
    m.encoder.to_mean.weight.requires_grad_(False)
    m.encoder.to_mean.bias.requires_grad_(False)

    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        train(m, data(30), config())

    text = " ".join(str(w.message) for w in caught)
    assert "KL warm-up" in text or "warm-up" in text
    assert "learning rate" in text


def test_the_collapse_is_reported_once_not_every_epoch():
    """Otherwise a long run buries everything else in the same warning."""
    import warnings as warnings_module

    import torch as torch_module

    m = model()
    with torch_module.no_grad():
        m.encoder.to_mean.weight.zero_()
        m.encoder.to_mean.bias.zero_()
    m.encoder.to_mean.weight.requires_grad_(False)
    m.encoder.to_mean.bias.requires_grad_(False)

    cfg = config().replace(train=TrainConfig(epochs=5, batch_size=10))
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        train(m, data(30), cfg)

    collapse_warnings = [w for w in caught if "latent space collapsed" in str(w.message)]
    assert len(collapse_warnings) == 1, f"warned {len(collapse_warnings)} times"


# --------------------------------------------------------------------------
# The device setting is acted on. Finding 45.
#
# RuntimeConfig validated the device string and its docstring said the device
# was "passed down". Nothing moved anything, so device="cuda" trained on the
# CPU without a word. Two tests here run everywhere and refuse a device the
# machine lacks; the gpu-marked one asserts the model actually went there.
# --------------------------------------------------------------------------


def _tiny_run(device: str, epochs: int = 2):
    """A 24-by-12 simple model trained on ``device``; returns model, history."""
    from pathlib import Path as _Path

    from tests.conftest import make_counts

    from preffect.config import (
        DataConfig,
        ModelConfig,
        ModelKind,
        PreffectConfig,
        RuntimeConfig,
        TrainConfig,
    )
    from preffect.data.tensors import to_tensors
    from preffect.model import SimpleModel

    cfg = PreffectConfig(
        data=DataConfig(train_path=_Path(".")),
        model=ModelConfig(kind=ModelKind.SIMPLE, latent_dim=4, hidden_dim=8),
        train=TrainConfig(epochs=epochs, batch_size=8),
        runtime=RuntimeConfig(seed=0, device=device, num_threads=2),
    )
    tensors = to_tensors(make_counts(24, 12, seed=0), cfg.model.correction_variables)
    net = SimpleModel(12, cfg.model)
    return net, train(net, tensors, cfg)


def test_a_cuda_device_is_refused_by_name_when_there_is_none(monkeypatch):
    import torch

    from preffect.config import RuntimeConfig
    from preffect.training import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ConfigError, match="no usable CUDA"):
        resolve_device(RuntimeConfig(device="cuda"))


def test_a_cuda_index_beyond_the_cards_present_is_refused(monkeypatch):
    import torch

    from preffect.config import RuntimeConfig
    from preffect.training import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ConfigError, match="2 CUDA device"):
        resolve_device(RuntimeConfig(device="cuda:5"))


def test_training_on_the_cpu_leaves_everything_on_the_cpu():
    net, history = _tiny_run("cpu")
    assert next(net.parameters()).device.type == "cpu"
    assert len(history) == 2


@pytest.mark.gpu
def test_training_runs_on_the_requested_device(gpu_device):
    """Before finding 45 this passed only in the sense that nothing raised."""
    import math

    net, history = _tiny_run(gpu_device)
    assert next(net.parameters()).device.type == "cuda"
    assert all(math.isfinite(x) for x in history.train_losses)


@pytest.mark.gpu
def test_a_graph_model_trains_on_the_requested_device(gpu_device):
    """The simple-model GPU test could not catch this: it has no edges to induce.

    Until finding 45's second half, the first batch of a single or full model
    on a GPU raised a device mismatch in the subgraph induction.
    """
    import math
    from pathlib import Path as _Path

    import numpy as np
    from tests.conftest import make_counts

    from preffect.config import (
        DataConfig,
        ModelConfig,
        ModelKind,
        PreffectConfig,
        RuntimeConfig,
        TrainConfig,
    )
    from preffect.data.tensors import to_tensors

    pytest.importorskip("torch_geometric", reason="needs the graph extra")
    from preffect.graph_model import SingleModel

    adata = make_counts(24, 12, seed=0)
    n = adata.n_obs
    ring = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        ring[i, (i + 1) % n] = ring[(i + 1) % n, i] = 1
    adata.obsm["sample_sample_adj"] = ring

    cfg = PreffectConfig(
        data=DataConfig(train_path=_Path(".")),
        model=ModelConfig(kind=ModelKind.SINGLE, latent_dim=4, hidden_dim=12),
        train=TrainConfig(epochs=2, batch_size=8),
        runtime=RuntimeConfig(seed=0, device=gpu_device, num_threads=2),
    )
    tensors = to_tensors(adata, cfg.model.correction_variables, with_network=True)
    net = SingleModel(12, cfg.model)
    history = train(net, tensors, cfg)
    assert next(net.parameters()).device.type == "cuda"
    assert all(math.isfinite(x) for x in history.train_losses)


# --------------------------------------------------------------------------
# deterministic=True makes a GPU run repeatable bit for bit. Issue #9.
#
# Measured before this existed: the simple model was already repeatable on a
# GPU with a seed; the graph model differed in 38 of 38 parameter tensors
# between two seeded runs, at the fourth decimal place, because the attention
# layer's scatter-add sums in whichever order the card finishes. With the four
# settings configure_runtime now applies, 0 of 38. Decision record
# 2026-10-27_gpu_runs_are_deterministic_when_asked.md.
# --------------------------------------------------------------------------


def _ring_counts(n_samples: int = 24, n_genes: int = 12):
    """Counts with a ring network, so a graph model has edges to attend to."""
    import numpy as np
    from tests.conftest import make_counts

    adata = make_counts(n_samples, n_genes, seed=0)
    ring = np.zeros((n_samples, n_samples), dtype=np.int64)
    for i in range(n_samples):
        ring[i, (i + 1) % n_samples] = ring[(i + 1) % n_samples, i] = 1
    adata.obsm["sample_sample_adj"] = ring
    return adata


def test_deterministic_applies_the_gpu_settings(monkeypatch):
    """Checkable without a GPU: the switches are global PyTorch state."""
    import os

    import torch

    from preffect.config import RuntimeConfig
    from preffect.training import configure_runtime

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    before = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        configure_runtime(config().replace(runtime=RuntimeConfig(seed=0, deterministic=True)))
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.backends.cudnn.benchmark = before[0]
        torch.backends.cudnn.deterministic = before[1]
        torch.use_deterministic_algorithms(before[2], warn_only=before[3])


def test_a_workspace_setting_already_in_the_environment_wins(monkeypatch):
    import os

    import torch

    from preffect.config import RuntimeConfig
    from preffect.training import configure_runtime

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        configure_runtime(config().replace(runtime=RuntimeConfig(seed=0, deterministic=True)))
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    finally:
        torch.use_deterministic_algorithms(before[0], warn_only=before[1])


@pytest.mark.gpu
def test_deterministic_makes_a_graph_model_repeatable_on_a_gpu(gpu_device):
    """The measurement that settled issue #9, as a test. Seeds before building."""
    from pathlib import Path as _Path

    import torch

    from preffect.config import (
        DataConfig,
        ModelConfig,
        ModelKind,
        PreffectConfig,
        RuntimeConfig,
        TrainConfig,
    )
    from preffect.data.tensors import to_tensors
    from preffect.training import configure_runtime

    pytest.importorskip("torch_geometric", reason="needs the graph extra")
    from preffect.graph_model import SingleModel

    cfg = PreffectConfig(
        data=DataConfig(train_path=_Path(".")),
        model=ModelConfig(kind=ModelKind.SINGLE, latent_dim=4, hidden_dim=12),
        train=TrainConfig(epochs=4, batch_size=8),
        runtime=RuntimeConfig(seed=3, device=gpu_device, num_threads=2, deterministic=True),
    )
    adata = _ring_counts()

    def once():
        tensors = to_tensors(adata, cfg.model.correction_variables, with_network=True)
        configure_runtime(cfg)
        net = SingleModel(12, cfg.model)
        train(net, tensors, cfg)
        return {k: v.detach().cpu() for k, v in net.state_dict().items()}

    first, second = once(), once()
    differing = [k for k in first if not torch.equal(first[k], second[k])]
    assert differing == [], f"{len(differing)} of {len(first)} tensors differ: {differing[:3]}"


# --------------------------------------------------------------------------
# The thread count has to be set before the model is built. Issue #23.
#
# Weights are drawn at construction, at whatever thread count is in force, and
# a model built at one count and trained at another is not the model a second
# call in the same process would build. On this objective the last-bit
# difference grew into a dead latent space on the first call of a script and a
# healthy one on every call after. train cannot re-draw anything, so it warns.
# The suite runs at two threads (conftest), so asking for four is the change.
# --------------------------------------------------------------------------


def _threads_config(n: int) -> PreffectConfig:
    return config().replace(runtime=RuntimeConfig(seed=0, num_threads=n))


def test_training_warns_when_the_model_was_built_at_another_thread_count():

    before = torch.get_num_threads()
    try:
        net = model()  # built at the suite's count
        with pytest.warns(RuntimeWarning, match="built while PyTorch was using"):
            train(net, data(30), _threads_config(before + 2))
    finally:
        torch.set_num_threads(before)


def test_configuring_before_building_is_silent_and_repeatable():
    """The documented order. No warning, and two runs agree exactly."""
    import warnings

    from preffect.training import configure_runtime

    before = torch.get_num_threads()
    cfg = _threads_config(before + 2)
    try:
        losses = []
        for _ in range(2):
            configure_runtime(cfg)
            net = model()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                history = train(net, data(30), cfg)
            assert not [w for w in caught if "thread" in str(w.message)], caught
            losses.append(history.train_losses)
        assert losses[0] == losses[1]
    finally:
        torch.set_num_threads(before)


def test_no_warning_when_no_thread_count_is_asked_for():
    import warnings

    net = model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train(net, data(30), config().replace(runtime=RuntimeConfig(seed=0)))
    assert not [w for w in caught if "thread" in str(w.message)]


# --------------------------------------------------------------------------
# What the optimiser saw and did, recorded per epoch. Issue #22, Phase 0.
#
# The excursion in issue #22 had only ever been seen as a loss sampled every
# 800 epochs. These three numbers are what tell a gradient that blew up from a
# loss that jumped on ordinary gradients, which point at different causes.
# --------------------------------------------------------------------------


def test_the_gradient_norm_is_recorded_before_clipping():
    """With the clip set far below any real gradient, the recorded norm is the
    one found before scaling, which is the number that says a gradient blew up."""
    import math

    cfg = config().replace(
        train=TrainConfig(epochs=1, batch_size=10, gradient_clip_norm=1e-6)
    )
    record = train(model(), data(30), cfg).epochs[0].train
    assert math.isfinite(record.largest_gradient_norm)
    assert record.largest_gradient_norm > 1e-3, "that is the clipped value, not the one before"
    assert 0 < record.mean_gradient_norm <= record.largest_gradient_norm


def test_the_largest_update_tracks_the_learning_rate():
    """Adam's step scales with the learning rate, so a tiny rate makes a tiny update."""
    small = train(model(), data(30), config().replace(
        train=TrainConfig(epochs=1, batch_size=10, learning_rate=1e-12)
    )).epochs[0].train.largest_update
    usual = train(model(), data(30), config().replace(
        train=TrainConfig(epochs=1, batch_size=10, learning_rate=1e-3)
    )).epochs[0].train.largest_update
    assert small < 1e-9, small
    assert usual > 1e-5, usual


def test_a_measuring_pass_records_no_gradient_or_update():
    result = run_epoch(model(), data(30), config(), epoch=1)
    assert result.largest_gradient_norm == 0.0
    assert result.mean_gradient_norm == 0.0
    assert result.largest_update == 0.0


# --------------------------------------------------------------------------
# Early stopping on the imputation error.
#
# Finding 50: the loss bottoms out about ten times sooner than imputation of
# the fitted cohort settles, so stopping on the loss keeps a model with twice
# the imputation error it would reach. The rule that watches the error on the
# hidden entries instead is issue #28; these say what it does.
# --------------------------------------------------------------------------


def masked_data(n_samples: int = 30, seed: int = 0) -> TensorData:
    from preffect.data.masking import apply_mask, mcar_mask

    full = data(n_samples, seed)
    counts = full.targets.numpy()
    mask = mcar_mask(counts.shape[0], counts.shape[1], 0.2, np.random.default_rng(seed))
    return TensorData(
        inputs=torch.from_numpy(apply_mask(counts, mask)), targets=full.targets, mask=mask
    )


def test_stopping_on_imputation_needs_hidden_entries():
    cfg = config().replace(
        train=TrainConfig(epochs=3, batch_size=10, early_stopping_patience=2, stop_on="imputation")
    )
    with pytest.raises(ConfigError, match="hides none"):
        train(model(), data(30), cfg)


def test_stopping_on_imputation_records_the_error_every_epoch():
    cfg = config().replace(
        train=TrainConfig(epochs=6, batch_size=10, early_stopping_patience=10, stop_on="imputation")
    )
    history = train(model(), masked_data(30), cfg)
    assert history.watched == "imputation"
    assert not history.watched_training_loss
    errors = [e.train.imputation_error for e in history.epochs]
    assert all(err is not None and np.isfinite(err) for err in errors)
    assert history.best_epoch == 1 + int(np.argmin(errors))


def test_stopping_on_imputation_keeps_the_best_epochs_model():
    """What is written is the model with the lowest error, not the last one."""
    from preffect.training import hidden_entry_error

    cfg = config().replace(
        train=TrainConfig(
            epochs=30,
            batch_size=10,
            learning_rate=1e-9,                # nothing improves after epoch 1
            early_stopping_patience=2,
            early_stopping_min_delta=1e-3,
            stop_on="imputation",
        )
    )
    m = model()
    history = train(m, masked_data(30), cfg)
    assert history.stopped_early and history.best_epoch == 1
    kept = hidden_entry_error(m, masked_data(30))
    assert kept == pytest.approx(history.epochs[0].train.imputation_error, rel=1e-6)


def test_stopping_on_the_loss_is_unchanged_by_the_new_rule():
    cfg = config().replace(train=TrainConfig(epochs=3, batch_size=10, early_stopping_patience=2))
    history = train(model(), masked_data(30), cfg)
    assert history.watched == "loss"
    assert all(e.train.imputation_error is None for e in history.epochs)


def test_measuring_the_error_leaves_the_run_itself_unchanged():
    """The measuring pass draws random numbers; the generator is put back, so the weights match."""
    plain = config().replace(train=TrainConfig(epochs=4, batch_size=10))
    watched = config().replace(
        train=TrainConfig(epochs=4, batch_size=10, early_stopping_patience=10, stop_on="imputation")
    )
    first, second = model(), model()
    train(first, masked_data(30), plain)
    train(second, masked_data(30), watched)
    pairs = zip(first.state_dict().items(), second.state_dict().items(), strict=True)
    for (name, a), (_, b) in pairs:
        assert torch.equal(a, b), f"{name} differs: measuring the error changed the run"



# --------------------------------------------------------------------------
# Every batch of a masked split carries the library-size correction.
#
# Finding 56: a batch has no mask of its own, because ``select`` drops it, so
# the correction travels as a field rather than being recomputed. If it did
# not reach the model, a mini-batched run would still be fitted toward a total
# it cannot reach even though inference on the whole split was right.
# --------------------------------------------------------------------------


class _RecordsItsScale(SimpleModel):
    """A model that remembers the correction each batch handed it."""

    def forward(self, counts, corrections=None, library_scale=None):
        self.seen.append(None if library_scale is None else library_scale.clone())
        return super().forward(counts, corrections, library_scale)


def _recording_model() -> _RecordsItsScale:
    torch.manual_seed(0)
    built = _RecordsItsScale(N_GENES, ModelConfig(hidden_dim=16, latent_dim=6))
    built.seen = []
    return built


def test_every_batch_of_a_masked_split_is_handed_its_own_correction():
    spy = _recording_model()
    split = masked_data(30)
    run_epoch(spy, split, config(), epoch=1)

    assert len(spy.seen) == 3, "three batches of ten"
    assert all(s is not None for s in spy.seen), "a batch reached the model uncorrected"
    assert sum(s.shape[0] for s in spy.seen) == 30
    # Every value is one of the split's own per-sample factors.
    whole = set(np.round(split.library_scale.numpy().ravel(), 6).tolist())
    for batch in spy.seen:
        assert set(np.round(batch.numpy().ravel(), 6).tolist()) <= whole


def test_a_split_with_nothing_hidden_hands_the_model_nothing():
    spy = _recording_model()
    run_epoch(spy, data(30), config(), epoch=1)
    assert spy.seen == [None, None, None]


def test_the_correction_changes_what_the_epoch_measures():
    """Otherwise it is being threaded through and dropped."""
    split = masked_data(30)
    uncorrected = TensorData(inputs=split.inputs.clone(), targets=split.targets)
    assert uncorrected.library_scale is None

    torch.manual_seed(0)
    with_it = run_epoch(model(), split, config(), epoch=1).mean_loss
    torch.manual_seed(0)
    without = run_epoch(model(), uncorrected, config(), epoch=1).mean_loss
    assert with_it != without
