"""Saving and reloading a model.

The point of this module is that a checkpoint holds tensors and settings, not a
pickled Python object. Version 0.1 pickled the whole live session, which
PyTorch 2.6 refuses to load, and which carried the training data and every
unused layer along with it.
"""

from __future__ import annotations

import pytest
import torch

from preffect import __version__
from preffect.checkpoint import (
    CHECKPOINT_FORMAT,
    load_checkpoint,
    save_checkpoint,
)
from preffect.config import CorrectionVariable, Likelihood, ModelConfig
from preffect.errors import CheckpointError
from preffect.model import SimpleModel

N_GENES = 15


class PretendVersion01Session:
    """Stands in for what version 0.1 wrote: a whole live object, pickled.

    Defined at module level because a class defined inside a function cannot be
    pickled, which is itself a hint about how fragile that storage format is.
    """

    def __init__(self):
        self.model = torch.nn.Linear(2, 2)
        self.train_dataset = [1, 2, 3]


def build(**changes) -> tuple[SimpleModel, ModelConfig]:
    config = ModelConfig(hidden_dim=12, latent_dim=5, **changes)
    torch.manual_seed(0)
    return SimpleModel(N_GENES, config), config


# --------------------------------------------------------------------------
# A checkpoint is tensors and settings.
# --------------------------------------------------------------------------


def test_a_saved_model_comes_back_the_same(tmp_path):
    model, config = build()
    path = save_checkpoint(tmp_path / "m.pt", model, config, N_GENES)

    loaded = load_checkpoint(path)
    rebuilt = loaded.build_model()

    for (name, original), (_, restored) in zip(
        model.state_dict().items(), rebuilt.state_dict().items(), strict=True
    ):
        torch.testing.assert_close(original, restored, msg=f"{name} differs")


def test_a_rebuilt_model_produces_the_same_output(tmp_path):
    model, config = build()
    model.eval()
    counts = torch.randint(0, 50, (6, N_GENES)).float()

    path = save_checkpoint(tmp_path / "m.pt", model, config, N_GENES)
    rebuilt = load_checkpoint(path).build_model()
    rebuilt.eval()

    torch.manual_seed(7)
    before = model(counts).composition
    torch.manual_seed(7)
    after = rebuilt(counts).composition
    torch.testing.assert_close(before, after)


def test_the_settings_travel_with_the_weights(tmp_path):
    model, config = build(
        likelihood=Likelihood.ZINB,
        infer_library_size=True,
        correction_variables=(CorrectionVariable("batch", "categorical"),),
    )
    path = save_checkpoint(tmp_path / "m.pt", model, config, N_GENES)
    loaded = load_checkpoint(path)

    assert loaded.model_config.likelihood is Likelihood.ZINB
    assert loaded.model_config.infer_library_size is True
    assert loaded.model_config.correction_variables == (
        CorrectionVariable("batch", "categorical"),
    )
    assert loaded.model_config.hidden_dim == 12


def test_gene_names_travel(tmp_path):
    """Without them nothing can check that new data has its columns in order."""
    model, config = build()
    names = tuple(f"Gene_{i}" for i in range(N_GENES))
    path = save_checkpoint(tmp_path / "m.pt", model, config, N_GENES, gene_names=names)
    assert load_checkpoint(path).gene_names == names


def test_the_optimiser_state_can_travel_too(tmp_path):
    model, config = build()
    optimiser = torch.optim.Adam(model.parameters())
    counts = torch.randint(0, 50, (6, N_GENES)).float()
    model(counts).composition.sum().backward()
    optimiser.step()

    path = save_checkpoint(
        tmp_path / "m.pt", model, config, N_GENES, optimiser=optimiser
    )
    assert load_checkpoint(path).optimiser_state is not None


def test_the_writing_version_is_recorded(tmp_path):
    model, config = build()
    path = save_checkpoint(tmp_path / "m.pt", model, config, N_GENES)
    loaded = load_checkpoint(path)
    assert loaded.written_by == __version__
    assert loaded.format_version == CHECKPOINT_FORMAT


def test_the_training_data_is_not_in_the_file(tmp_path):
    """Version 0.1 pickled the datasets along with the model."""
    model, config = build()
    path = save_checkpoint(tmp_path / "m.pt", model, config, N_GENES)

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload) == {
        "format_version", "written_by", "model_state", "model_config",
        "n_genes", "gene_names", "optimiser_state", "full_config", "extra",
    }
    # Everything stored is either a tensor of model weights or a plain value.
    assert all(isinstance(v, torch.Tensor) for v in payload["model_state"].values())


# --------------------------------------------------------------------------
# It loads under PyTorch's safe default, and says so when it cannot.
# --------------------------------------------------------------------------


def test_it_loads_with_the_safe_default():
    """If this stopped being true, version 0.2 would have version 0.1's problem."""
    import inspect

    from preffect import checkpoint

    source = inspect.getsource(checkpoint.load_checkpoint)
    assert "weights_only=True" in source


def test_a_version_0_1_style_pickled_session_is_refused_with_advice(tmp_path):
    path = tmp_path / "old_session.pth"
    torch.save(PretendVersion01Session(), path)

    with pytest.raises(CheckpointError) as excinfo:
        load_checkpoint(path)

    message = str(excinfo.value)
    assert "PREFFECT 0.1" in message
    assert "pickled" in message
    assert "Retrain" in message


def test_a_file_that_is_not_a_checkpoint_is_refused(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"something": torch.ones(3)}, path)
    with pytest.raises(CheckpointError, match="no format version"):
        load_checkpoint(path)


def test_an_unknown_format_version_is_refused(tmp_path):
    model, config = build()
    path = save_checkpoint(tmp_path / "m.pt", model, config, N_GENES)

    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["format_version"] = CHECKPOINT_FORMAT + 99
    torch.save(payload, path)

    with pytest.raises(CheckpointError, match="checkpoint format"):
        load_checkpoint(path)


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(CheckpointError, match="no checkpoint at"):
        load_checkpoint(tmp_path / "absent.pt")


def test_the_folder_is_created_if_it_does_not_exist(tmp_path):
    model, config = build()
    path = save_checkpoint(tmp_path / "deep" / "nested" / "m.pt", model, config, N_GENES)
    assert path.is_file()


# --------------------------------------------------------------------------
# A full model comes back as a full model.
#
# Finding 55: nothing had ever trained one, so nothing had ever reloaded one.
# ``build_model`` made a simple model whatever the settings said, which for a
# full model's weights is a PyTorch error about unexpected keys rather than a
# sentence.
# --------------------------------------------------------------------------


def two_tissue_model(n_tissues: int = 2):
    from preffect.config import ModelKind
    from preffect.full_model import FullModel

    config = ModelConfig(
        kind=ModelKind.FULL, hidden_dim=12, latent_dim=5, attention_heads=2
    )
    torch.manual_seed(0)
    return FullModel(N_GENES, config, n_tissues=n_tissues), config


def test_a_saved_full_model_is_rebuilt_as_one(tmp_path):
    pytest.importorskip("torch_geometric", reason="needs the graph extra")

    model, config = two_tissue_model()
    path = save_checkpoint(tmp_path / "full.pt", model, config, N_GENES, n_tissues=2)

    loaded = load_checkpoint(path)
    assert loaded.model_config.kind.value == "full"
    assert loaded.extra["n_tissues"] == 2

    rebuilt = loaded.build_model()
    assert type(rebuilt).__name__ == "FullModel"
    assert rebuilt.n_tissues == 2
    for (name, original), (_, restored) in zip(
        model.state_dict().items(), rebuilt.state_dict().items(), strict=True
    ):
        torch.testing.assert_close(original, restored, msg=f"{name} differs")


def test_a_rebuilt_full_model_produces_the_same_output(tmp_path):
    pytest.importorskip("torch_geometric", reason="needs the graph extra")

    model, config = two_tissue_model()
    model.eval()
    n_samples = 8
    counts = [torch.randint(0, 50, (n_samples, N_GENES)).float() for _ in range(2)]
    pairs = [(i, (i + 1) % n_samples) for i in range(n_samples)]
    edge = torch.tensor(pairs + [(b, a) for a, b in pairs], dtype=torch.long).t()

    path = save_checkpoint(tmp_path / "full.pt", model, config, N_GENES, n_tissues=2)
    rebuilt = load_checkpoint(path).build_model()
    rebuilt.eval()

    torch.manual_seed(7)
    before = model(counts, None, [edge, edge]).composition
    torch.manual_seed(7)
    after = rebuilt(counts, None, [edge, edge]).composition
    torch.testing.assert_close(before, after)


def test_a_full_model_saved_without_its_tissue_count_is_refused(tmp_path):
    """Rather than quietly rebuilt as a two-tissue model that is not it."""
    pytest.importorskip("torch_geometric", reason="needs the graph extra")

    model, config = two_tissue_model()
    path = save_checkpoint(tmp_path / "full.pt", model, config, N_GENES)
    with pytest.raises(CheckpointError, match="records 1 tissue"):
        load_checkpoint(path).build_model()


@pytest.mark.parametrize("kind", ["simple", "single", "full"])
def test_a_checkpoint_rebuilds_the_kind_of_model_it_holds(kind, tmp_path):
    """One round trip per kind, asserting the class that comes back.

    Finding 59: ``build_model`` built a simple model whatever the settings
    recorded, so a saved single or full model came back as the wrong class.
    Two of the three kinds were never round-tripped, which is how it lasted.
    """
    from preffect.config import ModelKind
    from preffect.workflow import build_model as build_kind

    if kind != "simple":
        pytest.importorskip("torch_geometric", reason="needs the graph extra")

    n_tissues = 2 if kind == "full" else 1
    config = ModelConfig(
        kind=ModelKind(kind), hidden_dim=12, latent_dim=5, attention_heads=2
    )
    torch.manual_seed(0)
    model = build_kind(ModelKind(kind), N_GENES, config, n_tissues)
    expected = type(model).__name__

    path = save_checkpoint(
        tmp_path / f"{kind}.pt", model, config, N_GENES, n_tissues=n_tissues
    )
    rebuilt = load_checkpoint(path).build_model()

    assert type(rebuilt).__name__ == expected, (
        f"a {kind} model was saved and came back as a {type(rebuilt).__name__}"
    )
    for (name, original), (_, restored) in zip(
        model.state_dict().items(), rebuilt.state_dict().items(), strict=True
    ):
        torch.testing.assert_close(original, restored, msg=f"{name} differs")


# --------------------------------------------------------------------------
# A saved model says how the run that produced it went.
#
# A checkpoint said what the model was and nothing about how it came to be, so
# the two questions anyone asks of a fit, how many epochs it took and whether
# it was stopped early, could only be answered from the terminal the run was
# started in. The record travels in ``extra``, which is free-form, so the
# format version does not move and a file written before it was kept still
# loads with nothing under the key.
# --------------------------------------------------------------------------


def a_history(n_epochs: int = 3, validation: bool = True, **changes):
    """A training history with plausible numbers, built without fitting."""
    from preffect.training import EpochRecord, SplitResult, TrainingHistory

    def split(loss: float) -> SplitResult:
        return SplitResult(
            mean_loss=loss, mean_terms={"expression": loss}, n_batches=3, n_samples=24
        )

    history = TrainingHistory(**changes)
    for epoch in range(1, n_epochs + 1):
        history.epochs.append(
            EpochRecord(
                epoch=epoch,
                train=split(100.0 - epoch),
                validation=split(110.0 - epoch) if validation else None,
            )
        )
    return history


def test_a_run_that_finished_records_every_epoch(tmp_path):
    from preffect.checkpoint import TRAINING_HISTORY_KEY, training_history_record

    model, config = build()
    record = training_history_record(
        a_history(n_epochs=3), epochs_asked_for=3, wall_seconds=12.5
    )
    path = save_checkpoint(
        tmp_path / "m.pt", model, config, N_GENES, **{TRAINING_HISTORY_KEY: record}
    )

    back = load_checkpoint(path).training
    assert back is not None
    assert back.epochs == (1, 2, 3)
    assert back.train_loss == (99.0, 98.0, 97.0)
    assert back.validation_loss == (109.0, 108.0, 107.0)
    assert back.n_epochs == 3 and back.epochs_asked_for == 3
    assert back.stopped_early is False
    assert back.wall_seconds == pytest.approx(12.5)
    assert "ran all 3 epoch(s) asked for" in back.stop_reason


def test_a_run_without_a_validation_split_records_one_curve(tmp_path):
    from preffect.checkpoint import TRAINING_HISTORY_KEY, training_history_record

    model, config = build()
    record = training_history_record(
        a_history(n_epochs=2, validation=False), epochs_asked_for=2, wall_seconds=1.0
    )
    path = save_checkpoint(
        tmp_path / "m.pt", model, config, N_GENES, **{TRAINING_HISTORY_KEY: record}
    )
    assert load_checkpoint(path).training.validation_loss is None


def test_an_early_stop_says_which_epoch_the_weights_came_from(tmp_path):
    """After an early stop the weights saved are the best epoch's, not the
    last, and a reader looking at a loss curve needs to be told which."""
    from preffect.checkpoint import TRAINING_HISTORY_KEY, training_history_record

    model, config = build()
    history = a_history(n_epochs=4, stopped_early=True, best_epoch=2)
    record = training_history_record(
        history, epochs_asked_for=50, wall_seconds=3.0, patience=2
    )
    path = save_checkpoint(
        tmp_path / "m.pt", model, config, N_GENES, **{TRAINING_HISTORY_KEY: record}
    )

    back = load_checkpoint(path).training
    assert back.stopped_early is True
    assert back.best_epoch == 2
    assert back.n_epochs == 4 and back.epochs_asked_for == 50
    assert "stopped early at epoch 4 of the 50 asked for" in back.stop_reason
    assert "validation loss had not improved for 2 epoch(s)" in back.stop_reason
    assert "from epoch 2" in back.stop_reason


def test_stopping_on_the_training_loss_is_named_as_such(tmp_path):
    """Early stopping without a validation split watches the training loss,
    which is a weaker thing to stop on, so the sentence says which it was."""
    from preffect.checkpoint import training_history_record

    history = a_history(n_epochs=2, validation=False, watched_training_loss=True)
    record = training_history_record(
        history, epochs_asked_for=2, wall_seconds=1.0, patience=5
    )
    assert "training loss never went 5 epoch(s) without improving" in record["stop_reason"]


def test_a_checkpoint_without_a_history_says_so(tmp_path):
    """Every file written before the record was kept. ``None`` rather than an
    error: an older checkpoint is still a valid one, and the interface says in
    a sentence that the history is missing."""
    model, config = build()
    path = save_checkpoint(tmp_path / "m.pt", model, config, N_GENES)
    assert load_checkpoint(path).training is None


def test_the_settings_come_back_with_the_model(tmp_path):
    """``full_config`` was written for provenance and never read, so nothing
    could say what batch size or learning rate a saved model was fitted at."""
    from preffect.config import DataConfig, PreffectConfig, TrainConfig

    model, config = build()
    full = PreffectConfig(
        data=DataConfig(train_path=tmp_path / "train"),
        model=config,
        train=TrainConfig(epochs=7, batch_size=33, learning_rate=0.005),
    )
    path = save_checkpoint(tmp_path / "m.pt", model, config, N_GENES, full_config=full)

    loaded = load_checkpoint(path)
    assert loaded.full_config["train"]["batch_size"] == 33
    assert loaded.full_config["train"]["learning_rate"] == pytest.approx(0.005)


def test_a_model_saved_without_its_settings_has_none_rather_than_a_default(tmp_path):
    """A default nothing chose, shown as though it had been, is how version
    0.1's shipped paths reached a published figure."""
    model, config = build()
    path = save_checkpoint(tmp_path / "m.pt", model, config, N_GENES)
    assert load_checkpoint(path).full_config is None
