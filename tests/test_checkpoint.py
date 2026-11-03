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
