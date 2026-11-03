"""The full model: several matched tissues.

This is the model the published release cannot construct at all, because its
graph decoder is built without a required argument. Finding 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric", reason="needs the graph extra")

from preffect.config import (  # noqa: E402
    Likelihood,
    LossWeights,
    ModelConfig,
    ModelKind,
)
from preffect.data.tensors import TensorData  # noqa: E402
from preffect.data.tissues import MultiTissueData, align_tissues  # noqa: E402
from preffect.errors import ConfigError, DataError  # noqa: E402
from preffect.full_model import FullModel  # noqa: E402
from preffect.loss import compute_loss  # noqa: E402

N_GENES, N_SAMPLES, N_TISSUES = 14, 10, 2


def ring(n: int = N_SAMPLES) -> torch.Tensor:
    pairs = [(i, (i + 1) % n) for i in range(n)]
    return torch.tensor(pairs + [(b, a) for a, b in pairs], dtype=torch.long).t()


def build(n_tissues: int = N_TISSUES, **changes) -> FullModel:
    config = ModelConfig(
        kind=ModelKind.FULL, hidden_dim=16, latent_dim=5, attention_heads=2, **changes
    )
    torch.manual_seed(0)
    return FullModel(N_GENES, config, n_tissues=n_tissues)


def tissue_counts(n_tissues: int = N_TISSUES) -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randint(0, 60, (N_SAMPLES, N_GENES)).float() for _ in range(n_tissues)]


def multi(n_tissues: int = N_TISSUES) -> MultiTissueData:
    return MultiTissueData(
        [
            TensorData(inputs=x.clone(), targets=x, edge_index=ring())
            for x in tissue_counts(n_tissues)
        ]
    )


# --------------------------------------------------------------------------
# It can be built and run, which the published release cannot do.
#
# Finding 1: preffect/_model.py:774 builds the graph decoder without r_embed,
# so constructing a full model raises TypeError before any data is read.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_tissues", [2, 3, 4])
def test_the_full_model_builds_for_any_tissue_count(n_tissues):
    model = build(n_tissues)
    assert len(model.encoders) == n_tissues
    assert len(model.network_decoders) == n_tissues


def test_the_pass_produces_one_network_result_per_tissue():
    out = build(3)(tissue_counts(3), None, [ring()] * 3)
    assert out.n_tissues == 3
    assert len(out.predicted_network) == 3
    assert all(p.shape == (N_SAMPLES, N_SAMPLES) for p in out.predicted_network)


def test_only_the_first_tissue_is_predicted():
    """Documented behaviour, and narrower than "multiple profiles" suggests."""
    out = build()(tissue_counts(), None, [ring()] * 2)
    assert out.composition.shape == (N_SAMPLES, N_GENES)  # one tissue's worth
    assert out.counts.mu.shape == (N_SAMPLES, N_GENES)


def test_the_target_library_size_comes_from_the_first_tissue():
    counts = tissue_counts()
    out = build()(counts, None, [ring()] * 2)
    torch.testing.assert_close(out.library_size, counts[0].sum(dim=1, keepdim=True))


def test_every_tissue_reaches_the_prediction():
    """If a tissue could not move the answer, combining them would be pointless."""
    counts = tissue_counts()
    model = build()
    model.eval()

    torch.manual_seed(1)
    before = model(counts, None, [ring()] * 2).composition

    changed = [counts[0], counts[1] + 500.0]  # only the second tissue moves
    torch.manual_seed(1)
    after = model(changed, None, [ring()] * 2).composition

    assert not torch.allclose(before, after)


@pytest.mark.parametrize("likelihood", [Likelihood.NB, Likelihood.ZINB])
def test_gradients_reach_every_parameter(likelihood):
    model = build(likelihood=likelihood)
    counts = tissue_counts()
    out = model(counts, None, [ring()] * 2)
    loss = -out.counts.log_prob(counts[0]).sum() + sum(
        p.sum() for p in out.predicted_network
    )
    loss.backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


# --------------------------------------------------------------------------
# Refusals.
# --------------------------------------------------------------------------


def test_the_wrong_kind_is_refused():
    with pytest.raises(ConfigError, match="needs kind 'full'"):
        FullModel(N_GENES, ModelConfig(kind=ModelKind.SINGLE, hidden_dim=16, latent_dim=5), 2)


def test_one_tissue_is_refused():
    with pytest.raises(ConfigError, match="at least 2"):
        build(1)


def test_a_mismatched_tissue_count_is_refused():
    with pytest.raises(ConfigError, match="built for 2 tissues"):
        build(2)(tissue_counts(3), None, [ring()] * 3)


def test_missing_networks_are_refused():
    with pytest.raises(ConfigError, match="one sample-sample network per tissue"):
        build()(tissue_counts(), None, None)


# --------------------------------------------------------------------------
# One loss term per tissue, each with its own weight.
#
# Finding 4: version 0.1 paired a stack of per-tissue losses against the
# weight list with zip, which stops at the shorter, so a two-tissue run with a
# one-entry list dropped the second tissue from the objective.
# --------------------------------------------------------------------------


def test_each_tissue_gets_its_own_named_network_terms():
    data = multi(2)
    out = build()( [t.inputs for t in data.tissues], None, [t.edge_index for t in data.tissues])
    loss = compute_loss(
        out,
        data.target.targets,
        LossWeights().for_tissue_count(2),
        network=[t.adjacency_matrix() for t in data.tissues],
    )
    names = {t.name for t in loss.terms}
    assert {"adjacency_recon_1", "adjacency_recon_2"} <= names
    assert {"adjacency_kl_1", "adjacency_kl_2"} <= names


def test_a_short_weight_list_is_refused_rather_than_truncated():
    data = multi(2)
    out = build()([t.inputs for t in data.tissues], None, [t.edge_index for t in data.tissues])
    with pytest.raises(ConfigError, match="one entry per tissue"):
        compute_loss(
            out,
            data.target.targets,
            LossWeights(),  # length-one lists, two tissues
            network=[t.adjacency_matrix() for t in data.tissues],
        )


def test_every_tissue_contributes_to_the_divisor():
    data_two, data_three = multi(2), multi(3)
    loss_two = compute_loss(
        build(2)([t.inputs for t in data_two.tissues], None,
                 [t.edge_index for t in data_two.tissues]),
        data_two.target.targets,
        LossWeights().for_tissue_count(2),
        network=[t.adjacency_matrix() for t in data_two.tissues],
    )
    loss_three = compute_loss(
        build(3)([t.inputs for t in data_three.tissues], None,
                 [t.edge_index for t in data_three.tissues]),
        data_three.target.targets,
        LossWeights().for_tissue_count(3),
        network=[t.adjacency_matrix() for t in data_three.tissues],
    )
    assert loss_three.weight_total > loss_two.weight_total


def test_a_delay_applies_to_every_tissue_of_a_term():
    from preffect.config import LossDelays

    delays = LossDelays(adjacency_recon=5)
    assert not delays.active_at("adjacency_recon_1", 1)
    assert not delays.active_at("adjacency_recon_2", 1)
    assert delays.active_at("adjacency_recon_2", 5)


# --------------------------------------------------------------------------
# Aligning tissues by their correspondence table.
# --------------------------------------------------------------------------


def paired_tissues(missing: tuple[int, ...] = (), network: bool = False):
    """Two tissues of four patients, the second scrambled so aligning has work to do.

    ``missing`` names the patients who have no sample in the second tissue.
    Their cell in the correspondence table is empty **and** their sample is
    absent from that tissue's file, which is the shape a real cohort takes: a
    patient with no matched normal has no normal profile to store.
    """
    import anndata as ad
    import pandas as pd

    names = [[f"tau{t}_{i}" for i in range(4)] for t in (1, 2)]
    table = pd.DataFrame({"tau_1": names[0], "tau_2": list(names[1])})
    for patient in missing:
        table.loc[patient, "tau_2"] = np.nan

    out = []
    for t, own in enumerate(names):
        rows = [i for i in range(4) if not (t == 1 and i in missing)]
        kept = [own[i] for i in rows]
        frame = table.iloc[rows].copy()
        frame.index = kept
        adata = ad.AnnData(
            X=(np.arange(4 * 3, dtype=np.float32).reshape(4, 3) + t * 100)[rows],
            obs=pd.DataFrame(index=kept),
        )
        adata.obsm["sample_association"] = frame
        if network:
            # A ring over whatever samples this tissue has, so every real
            # sample has two links and a padded one can be seen to have none.
            m = len(kept)
            ring_matrix = np.zeros((m, m), dtype=np.int64)
            for i in range(m):
                ring_matrix[i, (i + 1) % m] = ring_matrix[(i + 1) % m, i] = 1
            adata.obsm["sample_sample_adj"] = ring_matrix
        out.append(adata)
    # scramble the second tissue's row order, so alignment has work to do
    order = list(range(out[1].n_obs))[::-1]
    out[1] = out[1][order].copy()
    return out


def test_alignment_puts_matching_patients_on_the_same_row():
    aligned = align_tissues(paired_tissues()).tissues
    assert list(aligned[0].obs_names) == ["tau1_0", "tau1_1", "tau1_2", "tau1_3"]
    assert list(aligned[1].obs_names) == ["tau2_0", "tau2_1", "tau2_2", "tau2_3"]


def test_alignment_needs_the_correspondence_table():
    import anndata as ad
    import pandas as pd

    plain = [
        ad.AnnData(
            X=np.zeros((3, 2), dtype=np.float32),
            obs=pd.DataFrame(index=[f"s{i}" for i in range(3)]),
        )
        for _ in range(2)
    ]
    with pytest.raises(DataError, match="no sample correspondence table"):
        align_tissues(plain)


def test_one_tissue_needs_no_alignment():
    import anndata as ad
    import pandas as pd

    one = [ad.AnnData(X=np.zeros((3, 2), dtype=np.float32),
                      obs=pd.DataFrame(index=["a", "b", "c"]))]
    assert align_tissues(one).tissues == one


# --------------------------------------------------------------------------
# A correspondence table that does not name a tissue's samples once each.
#
# Finding 62: aligning on such a table used to lose samples without a word.
# A counterpart named for two patients made the row appear twice and dropped
# another sample entirely; a sample the table never named was dropped
# outright. Only an anndata warning about repeated names fired, and the run
# carried on over data that was not the data in the file.
# --------------------------------------------------------------------------


def test_a_counterpart_named_for_two_patients_is_refused():
    tissues = paired_tissues()
    for tissue in tissues:
        tissue.obsm["sample_association"].loc[:, "tau_2"] = [
            "tau2_0", "tau2_1", "tau2_1", "tau2_3"
        ][: tissue.n_obs]
    with pytest.raises(DataError, match="counterpart of more than one patient"):
        align_tissues(tissues)


def test_a_sample_the_table_never_names_is_refused():
    """The table in both files stops mentioning one patient, whose samples stay.

    Aligning used to return a cohort two samples smaller than the files, with
    nothing said.
    """
    import anndata as ad
    import pandas as pd

    names = [[f"tau{t}_{i}" for i in range(4)] for t in (1, 2)]
    table = pd.DataFrame({"tau_1": names[0][:3], "tau_2": names[1][:3]})
    tissues = []
    for own in names:
        frame = table.copy()
        # An obsm frame needs one row per sample, so the forgotten patient's
        # row repeats another patient's, which is how such a table arises.
        frame = pd.concat([frame, frame.iloc[[0]]])
        frame.index = own
        tissues.append(
            ad.AnnData(
                X=np.zeros((4, 3), dtype=np.float32),
                obs=pd.DataFrame(index=own),
            )
        )
        tissues[-1].obsm["sample_association"] = frame
    with pytest.raises(DataError, match="never names|more than one patient"):
        align_tissues(tissues)


# --------------------------------------------------------------------------
# Patients missing a tissue are padded, which is version 0.1's handling
# ported as it stands. Issue #14, decided 2026-11-02.
#
# One test per inherited rule: what a padded row's counts are, that it has no
# links, that the record of which rows are padding is returned per tissue and
# travels into a batch, and that the old refusal is still reachable because
# dropping the incomplete patients is the alternative being measured.
# --------------------------------------------------------------------------


def test_a_padded_row_carries_the_rounded_gene_mean_of_the_real_rows():
    from preffect.data.tissues import ghost_profile

    tissues = paired_tissues(missing=(2,))
    real = np.asarray(tissues[1].X, dtype=np.float32)
    aligned = align_tissues(tissues)
    padded = np.asarray(aligned.tissues[1].X, dtype=np.float32)[2]
    expected = np.round(real.mean(axis=0)).astype(np.float32)
    np.testing.assert_array_equal(padded, expected)
    np.testing.assert_array_equal(padded, ghost_profile(real))


def test_a_padded_row_has_no_links_in_the_network():
    aligned = align_tissues(paired_tissues(missing=(2,), network=True))
    network = np.asarray(aligned.tissues[1].obsm["sample_sample_adj"])
    assert network.shape == (4, 4)
    assert network[2].sum() == 0 and network[:, 2].sum() == 0
    assert network.sum() > 0, "the real samples should still be linked"


def test_alignment_records_which_rows_are_padding_in_each_tissue():
    aligned = align_tissues(paired_tissues(missing=(1, 2)))
    assert aligned.n_ghosts == [0, 2]
    np.testing.assert_array_equal(aligned.ghosts[0], [False] * 4)
    np.testing.assert_array_equal(aligned.ghosts[1], [False, True, True, False])


def test_a_padded_row_carries_no_annotations():
    """Every annotation missing, so nothing about it can be mistaken for measured."""
    import pandas as pd

    tissues = paired_tissues(missing=(2,))
    tissues[1].obs["subtype"] = pd.Categorical(["a", "b", "c"])
    aligned = align_tissues(tissues)
    assert pd.isna(aligned.tissues[1].obs["subtype"].iloc[2])
    assert aligned.tissues[1].obs["preffect_ghost"].tolist() == [0, 0, 1, 0]


def test_the_table_keeps_one_row_per_patient_with_the_padded_name_filled_in():
    aligned = align_tissues(paired_tissues(missing=(2,)))
    table = aligned.tissues[1].obsm["sample_association"]
    assert len(table) == 4
    assert not table.isna().to_numpy().any()
    assert table["tau_2"].iloc[2] == "tau_2_1"
    assert list(aligned.tissues[1].obs_names) == list(table["tau_2"])


def test_the_old_refusal_is_still_reachable_by_asking_for_it():
    """Dropping the incomplete patients is the alternative padding is measured against."""
    with pytest.raises(DataError, match="no counterpart"):
        align_tissues(paired_tissues(missing=(2,)), unmatched="refuse")


def test_a_complete_cohort_records_no_padding_and_is_not_rebuilt():
    aligned = align_tissues(paired_tissues())
    assert aligned.n_ghosts == [0, 0]
    assert not aligned.any_ghosts


def test_unmatched_counts_say_how_many_patients_lack_each_tissue():
    from preffect.data.tissues import unmatched_counts

    counts = unmatched_counts(paired_tissues(missing=(0, 3)))
    assert counts == {"tau_1": 0, "tau_2": 2}


def test_tissues_must_agree_on_their_sample_count():
    a = TensorData(inputs=torch.zeros(5, 4), targets=torch.zeros(5, 4))
    b = TensorData(inputs=torch.zeros(3, 4), targets=torch.zeros(3, 4))
    with pytest.raises(DataError, match="different sample counts"):
        MultiTissueData([a, b])


def test_selecting_a_batch_takes_the_same_rows_from_every_tissue():
    data = multi(3).select(np.array([0, 1, 2]))
    assert data.n_tissues == 3
    assert all(t.n_samples == 3 for t in data.tissues)


# --------------------------------------------------------------------------
# The library-size correction is the target tissue's.
#
# Finding 56. The full model predicts the first tissue's counts only, so the
# first tissue's total is the only one that sets a scale, and the only one a
# correction applies to.
# --------------------------------------------------------------------------


def test_the_target_library_is_scaled_by_what_the_pass_is_given():
    data = multi(2)
    scale = torch.full((N_SAMPLES, 1), 1.25)
    model = build()
    torch.manual_seed(0)
    plain = model([t.inputs for t in data.tissues], None,
                  [t.edge_index for t in data.tissues]).library_size
    torch.manual_seed(0)
    corrected = model([t.inputs for t in data.tissues], None,
                      [t.edge_index for t in data.tissues],
                      library_scale=scale).library_size
    torch.testing.assert_close(corrected, plain * 1.25)


def test_inference_takes_the_correction_from_the_target_tissue():
    """A second tissue's mask must not move the prediction's scale."""
    import numpy as np

    from preffect.data.masking import apply_mask, mcar_mask
    from preffect.inference import infer

    rng = np.random.default_rng(0)
    values = [rng.integers(5, 200, size=(N_SAMPLES, N_GENES)).astype(np.float32)
              for _ in range(2)]
    masks = [mcar_mask(N_SAMPLES, N_GENES, f, np.random.default_rng(seed))
             for seed, f in enumerate((0.2, 0.5))]
    data = MultiTissueData([
        TensorData(
            inputs=torch.from_numpy(apply_mask(v, m)),
            targets=torch.from_numpy(v),
            mask=m,
            edge_index=ring(),
        )
        for v, m in zip(values, masks, strict=True)
    ])

    result = infer(build(), data)
    expected = data.target.inputs.sum(dim=1, keepdim=True) * data.target.library_scale
    torch.testing.assert_close(result.library_size, expected)
    # The second tissue hides more than twice as much, so taking its factor
    # instead would be visibly different.
    wrong = data.tissues[1].library_scale
    assert not torch.allclose(data.target.library_scale, wrong)


# --------------------------------------------------------------------------
# The full model trains.
#
# Finding 55: ``run_epoch`` asked every batch whether it had a network, which
# a multi-tissue batch does not answer, so every full-model run stopped at its
# first batch with an AttributeError. Nothing caught it because the tests
# above build the model, run its forward pass and compute its loss, and stop
# one call short of ``train``. These call ``train``.
# --------------------------------------------------------------------------


def training_config(epochs: int = 8, n_tissues: int = N_TISSUES, **changes):
    from preffect.config import (
        DataConfig,
        PreffectConfig,
        RuntimeConfig,
        TrainConfig,
    )

    return PreffectConfig(
        data=DataConfig(train_path=Path("/nowhere"), n_tissues=n_tissues),
        model=ModelConfig(
            kind=ModelKind.FULL, hidden_dim=16, latent_dim=5, attention_heads=2
        ),
        loss=LossWeights().for_tissue_count(n_tissues),
        train=TrainConfig(epochs=epochs, batch_size=5, learning_rate=1e-2, **changes),
        runtime=RuntimeConfig(seed=0, num_threads=2),
    )


def test_a_two_tissue_model_trains_and_the_loss_falls():
    from preffect.training import train

    history = train(build(), multi(2), training_config())
    assert len(history) == 8
    losses = history.train_losses
    assert all(np.isfinite(v) for v in losses)
    assert losses[-1] < losses[0], losses


def test_training_runs_in_mini_batches_over_every_sample():
    """Ten samples at a batch of five is two batches, not one pass or none."""
    from preffect.training import run_epoch

    result = run_epoch(build(), multi(2), training_config(), epoch=1)
    assert result.n_batches == 2
    assert result.n_samples == N_SAMPLES


def test_the_objective_carries_one_network_term_per_tissue_while_training():
    from preffect.training import run_epoch

    terms = run_epoch(
        build(3), multi(3), training_config(n_tissues=3), epoch=1
    ).mean_terms
    assert {"adjacency_recon_1", "adjacency_recon_2", "adjacency_recon_3"} <= set(terms)
    assert {"adjacency_kl_1", "adjacency_kl_2", "adjacency_kl_3"} <= set(terms)
    assert "expression_recon" in terms


def test_the_weights_actually_move():
    """A loop that ran but trained nothing would pass the test above."""
    import copy

    from preffect.training import train

    model = build()
    before = copy.deepcopy(model.state_dict())
    train(model, multi(2), training_config())
    moved = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value, before[name])
    ]
    assert moved, "training left every parameter where it was"


def test_a_validation_split_is_scored_too():
    from preffect.training import train

    history = train(build(), multi(2), training_config(epochs=3), multi(2))
    assert all(e.validation is not None for e in history.epochs)
    assert len(history.validation_losses) == 3


def test_stopping_on_imputation_is_refused_for_matched_tissues_by_name():
    """Hiding entries across tissues is not built, and ``--hide`` says so too."""
    from preffect.errors import ConfigError
    from preffect.training import train

    config = training_config(epochs=3, early_stopping_patience=2, stop_on="imputation")
    with pytest.raises(ConfigError, match="one table per tissue"):
        train(build(), multi(2), config)


# --------------------------------------------------------------------------
# What a padded row contributes to the objective, and what it does not.
#
# Version 0.1 removed ghost rows from the expression reconstruction, the
# library size and its latent, and kept them in the latent and network terms:
# the lines that would have removed them from mu_X, logvar_X, mu_A and
# logvar_A are commented out in its own source. Ported as it stands and
# measured in exp/19. Issue #14, decided 2026-11-02.
# --------------------------------------------------------------------------


def padded_multi(n_padded: int = 3, n_tissues: int = N_TISSUES) -> MultiTissueData:
    """A bundle whose target tissue's last rows are padding."""
    ghost = torch.zeros(N_SAMPLES, dtype=torch.bool)
    if n_padded:
        ghost[N_SAMPLES - n_padded:] = True
    return MultiTissueData(
        [
            TensorData(inputs=x.clone(), targets=x, edge_index=ring())
            for x in tissue_counts(n_tissues)
        ],
        ghosts=[ghost.clone() for _ in range(n_tissues)],
    )


def loss_of(data: MultiTissueData, real_rows, model=None, **changes):
    model = model or build(data.n_tissues, **changes)
    out = model(
        [t.inputs for t in data.tissues], None, [t.edge_index for t in data.tissues]
    )
    return compute_loss(
        out,
        data.target.targets,
        LossWeights().for_tissue_count(data.n_tissues),
        network=[t.adjacency_matrix() for t in data.tissues],
        real_rows=real_rows,
    )


def test_only_the_terms_that_read_the_counts_leave_padded_rows_out():
    """The same pass scored twice: once over every row, once over the real ones."""
    data = padded_multi()
    model = build()
    torch.manual_seed(1)
    over_all = loss_of(data, None, model=model).named_values()
    torch.manual_seed(1)
    over_real = loss_of(data, data.target_real_rows(), model=model).named_values()

    assert over_all["expression_recon"] != over_real["expression_recon"]
    for name in ("expression_kl", "adjacency_recon_1", "adjacency_recon_2",
                 "adjacency_kl_1", "adjacency_kl_2"):
        assert over_all[name] == over_real[name], name


def test_the_library_terms_leave_padded_rows_out_too():
    data = padded_multi()
    model = build(infer_library_size=True)
    torch.manual_seed(1)
    over_all = loss_of(data, None, model=model).named_values()
    torch.manual_seed(1)
    over_real = loss_of(data, data.target_real_rows(), model=model).named_values()
    assert over_all["library_recon"] != over_real["library_recon"]
    assert over_all["library_kl"] != over_real["library_kl"]


def test_the_expression_term_ignores_what_a_padded_row_is_scored_against():
    """Changing a padded row's target must not move the term, which is the point."""
    data = padded_multi()
    model = build()
    real = data.target_real_rows()

    torch.manual_seed(1)
    before = loss_of(data, real, model=model).named_values()["expression_recon"]

    changed = [t for t in data.tissues]
    targets = changed[0].targets.clone()
    targets[~real] += 1000.0
    changed[0] = TensorData(
        inputs=changed[0].inputs, targets=targets, edge_index=changed[0].edge_index
    )
    moved = MultiTissueData(changed, ghosts=data.ghosts)
    torch.manual_seed(1)
    after = loss_of(moved, real, model=model).named_values()["expression_recon"]
    assert before == after


def test_a_batch_of_nothing_but_padding_is_refused():
    """Version 0.1 averaged over nothing here and produced a loss that was not a number."""
    data = padded_multi(n_padded=N_SAMPLES)
    with pytest.raises(ConfigError, match="every one of this batch"):
        loss_of(data, data.target_real_rows())


def test_a_bundle_with_nothing_padded_asks_for_no_selection():
    """None, not an all-true selection, so a complete cohort is scored as before."""
    assert multi(2).target_real_rows() is None
    assert padded_multi(n_padded=0).target_real_rows() is None


def test_the_record_of_padded_rows_follows_a_batch():
    batch = padded_multi().select(np.array([0, 1, N_SAMPLES - 1]))
    assert batch.n_ghosts == [1, 1]
    assert batch.target_real_rows().tolist() == [True, True, False]


def test_the_record_of_padded_rows_must_match_the_rows():
    with pytest.raises(DataError, match="covers 3"):
        MultiTissueData(multi(2).tissues, ghosts=[torch.zeros(3, dtype=torch.bool)] * 2)


def test_there_must_be_one_record_per_tissue():
    with pytest.raises(DataError, match="one per tissue"):
        MultiTissueData(multi(2).tissues, ghosts=[torch.zeros(N_SAMPLES, dtype=torch.bool)])


def test_a_mask_that_hides_an_entry_in_a_padded_row_is_refused():
    """Those counts are per-gene means, so recovering one measures nothing."""
    from preffect.data.masking import Mask, apply_mask

    counts = tissue_counts()
    hidden = np.zeros((N_SAMPLES, N_GENES), dtype=bool)
    hidden[N_SAMPLES - 1, :3] = True          # the last row is padding below
    mask = Mask(hidden=hidden)
    ghost = torch.zeros(N_SAMPLES, dtype=torch.bool)
    ghost[-1] = True
    parts = [
        TensorData(
            inputs=torch.from_numpy(apply_mask(x.numpy(), mask)),
            targets=x,
            mask=mask,
            edge_index=ring(),
        )
        for x in counts
    ]
    with pytest.raises(DataError, match="padded in for patients missing"):
        MultiTissueData(parts, ghosts=[ghost.clone(), ghost.clone()])


def test_a_mask_that_hides_nothing_in_a_padded_row_is_allowed():
    from preffect.data.masking import Mask, apply_mask

    counts = tissue_counts()
    hidden = np.zeros((N_SAMPLES, N_GENES), dtype=bool)
    hidden[0, :3] = True
    mask = Mask(hidden=hidden)
    ghost = torch.zeros(N_SAMPLES, dtype=torch.bool)
    ghost[-1] = True
    parts = [
        TensorData(
            inputs=torch.from_numpy(apply_mask(x.numpy(), mask)),
            targets=x,
            mask=mask,
            edge_index=ring(),
        )
        for x in counts
    ]
    data = MultiTissueData(parts, ghosts=[ghost.clone(), ghost.clone()])
    assert data.n_ghosts == [1, 1]


# --------------------------------------------------------------------------
# What comes out: padded rows are dropped and counted.
# --------------------------------------------------------------------------


def test_inference_drops_padded_rows_and_says_how_many():
    from preffect.inference import infer

    data = padded_multi(n_padded=3)
    result = infer(build(), data)
    assert result.n_padded_rows_dropped == 3
    assert result.n_samples == N_SAMPLES - 3
    assert result.latent.shape[0] == N_SAMPLES - 3
    assert result.composition.shape == (N_SAMPLES - 3, N_GENES)


def test_inference_on_a_complete_cohort_drops_nothing():
    from preffect.inference import infer

    result = infer(build(), multi(2))
    assert result.n_padded_rows_dropped == 0
    assert result.n_samples == N_SAMPLES


def test_a_padded_row_still_reaches_the_encoder():
    """It is a node of the network the real rows attend over, so it is not simply removed."""
    from preffect.inference import infer

    data = padded_multi(n_padded=3)
    model = build()
    model.eval()
    first = infer(model, data).expected_counts

    moved = [t for t in data.tissues]
    inputs = moved[0].inputs.clone()
    inputs[-3:] += 500.0            # change only the padded rows' counts
    moved[0] = TensorData(inputs=inputs, targets=moved[0].targets,
                          edge_index=moved[0].edge_index)
    second = infer(model, MultiTissueData(moved, ghosts=data.ghosts)).expected_counts
    assert not torch.allclose(first, second)


# --------------------------------------------------------------------------
# A cohort with gaps trains end to end, and one without is unchanged.
# --------------------------------------------------------------------------


def test_the_full_model_trains_on_a_cohort_with_gaps():
    from preffect.training import train

    history = train(build(), padded_multi(n_padded=3), training_config())
    assert len(history) == 8
    assert all(np.isfinite(v) for v in history.train_losses)
    assert history.train_losses[-1] < history.train_losses[0], history.train_losses


def test_a_cohort_with_nothing_padded_trains_exactly_as_it_did():
    """The padding seam must be invisible when there is nothing to pad."""
    import copy

    from preffect.training import train

    plain = build()
    before = copy.deepcopy(plain.state_dict())
    train(plain, multi(2), training_config())
    after_plain = copy.deepcopy(plain.state_dict())

    # The same data, carrying an explicit record that says nothing is padded.
    empty = MultiTissueData(
        multi(2).tissues,
        ghosts=[torch.zeros(N_SAMPLES, dtype=torch.bool) for _ in range(2)],
    )
    other = build()
    other.load_state_dict(before)
    train(other, empty, training_config())

    for name, value in other.state_dict().items():
        assert torch.equal(value, after_plain[name]), name
