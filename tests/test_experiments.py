"""Every experiment's arms must build different models.

Finding 27: `exp/03`'s baseline arm was plain `SimpleModel`, and acting on the
experiment's own result changed `SimpleModel`, so from that commit both arms
were the same model. It returned a perfect null, bit for bit, reported it as
"within noise", and a decision record cited it for a day.

`exp/common.py` now prints IDENTICAL when every measure matches exactly, but
that only helps someone who runs the experiment and reads the output. These
tests fail in the ordinary suite instead, which is the half that was missing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

EXPERIMENTS = Path(__file__).resolve().parents[1] / "exp"
if not EXPERIMENTS.is_dir():
    pytest.skip("exp/ is part of the development record, not of a release", allow_module_level=True)

#: Every experiment that builds more than one model in order to compare them,
#: and the name in it that this file checks. An experiment that compares two
#: models and is not listed here fails
#: :func:`test_every_experiment_with_arms_is_checked_here`.
#:
#: That test is the half of finding 27 that was still missing. The bug was not
#: only that `exp/03`'s arms became the same model; it was that nothing
#: required an experiment to be checked at all, so the next one would have gone
#: the same way. Listing them by hand would have the same weakness, which is
#: why the test looks at what is in `exp/` rather than at this dictionary.
ARMS = {
    "01_latent_rectifier_removal": "WithRectifier",
    "02_corrections_raw_or_embedded": "Embedded",
    "03_decoder_rectifier_removal": "TwoRectifiers",
    "05_rectifier_removals_at_full_size": "build_arms",
    "08_graph_model_whole_cohort_batch": "build_arms",
    "10_log_counts_at_the_encoder": "RawInputModel",
    "12_stopping_rule_live": "build_arms",
    "14_masking_zeros": "build_arms",
    "15_encoder_input_scaling": "build_arms",
    "16_cohort_sample_network": "build_arms",
    "17_network_term_weighting": "build_arms",
    "18_full_model_first_run": "build_arms",
    "19_unmatched_samples_padded": "build_arms",
}

#: Experiments that compare nothing, with the reason. Listed so that adding an
#: experiment forces a decision about which kind it is rather than letting it
#: go unchecked by default.
NO_ARMS = {
    "04_speed_and_memory_at_full_size": "measures one model's speed and memory at full size",
    "06_pam50_subtyping_replication": "reproduces one published analysis with one model",
    "07_kl_weight_vs_clustering": "sweeps a weight, so every run is the same model",
    "09_loss_spikes_after_convergence": "one model, three seeds, instruments on; traces, no arms",
    "11_training_length_and_stopping": "one model per seed, read at checkpoints; no arms",
    "13_versions_end_to_end": "two versions of the package on one protocol; no model arms",
}


def load(name: str):
    """Import an experiment's run_all.py without running it."""
    sys.path.insert(0, str(EXPERIMENTS))
    path = EXPERIMENTS / name / "run_all.py"
    if not path.exists():
        pytest.skip(f"{name} has no run_all.py")
    spec = importlib.util.spec_from_file_location(f"exp_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def differs(first, second, n_genes: int = 20) -> bool:
    """Do two models produce different output from the same input and seed?"""
    from preffect.config import ModelConfig

    config = ModelConfig(hidden_dim=16, latent_dim=4)
    torch.manual_seed(0)
    a = first(n_genes, config)
    torch.manual_seed(0)
    b = second(n_genes, config)

    counts = torch.randint(0, 50, (8, n_genes)).float()
    a.eval()
    b.eval()
    with torch.no_grad():
        torch.manual_seed(1)
        out_a = a(counts)
        torch.manual_seed(1)
        out_b = b(counts)
    return not torch.allclose(out_a.composition, out_b.composition)


@pytest.mark.parametrize(
    "experiment,builder",
    [
        ("01_latent_rectifier_removal", "WithRectifier"),
        ("03_decoder_rectifier_removal", "TwoRectifiers"),
        ("10_log_counts_at_the_encoder", "RawInputModel"),
    ],
)
def test_the_arm_under_test_differs_from_the_package_default(experiment, builder):
    """The exact failure of finding 27, as a test rather than a printed note."""
    from preffect.model import SimpleModel

    module = load(experiment)
    arm = getattr(module, builder, None)
    assert arm is not None, f"{experiment} no longer defines {builder}"

    assert differs(arm, SimpleModel), (
        f"{experiment}'s '{builder}' arm produces the same output as the "
        f"package default, so the experiment is comparing a model with itself"
    )


def test_exp_02_arms_differ():
    """exp/02 compares concatenated against embedded correction variables."""
    from preffect.config import CorrectionVariable, ModelConfig
    from preffect.model import SimpleModel

    module = load("02_corrections_raw_or_embedded")
    embedded = getattr(module, "Embedded", None)
    assert embedded is not None, "exp/02 no longer defines Embedded"

    config = ModelConfig(
        hidden_dim=16, latent_dim=4,
        correction_variables=(CorrectionVariable("batch", "categorical"),),
    )
    counts = torch.randint(0, 50, (8, 20)).float()
    corrections = torch.randint(0, 3, (8, 1)).float()

    torch.manual_seed(0)
    plain = SimpleModel(20, config).eval()
    torch.manual_seed(0)
    other = embedded(20, config).eval()
    with torch.no_grad():
        torch.manual_seed(1)
        a = plain(counts, corrections)
        torch.manual_seed(1)
        b = other(counts, corrections)
    assert not torch.allclose(a.composition, b.composition), (
        "exp/02's two arms produce identical output"
    )


@pytest.mark.parametrize("seam", ["latent", "decoded"])
def test_exp_05_arms_differ(seam):
    """exp/05 repeats the rectifier questions at full size, two arms each.

    Its 'removed' arm is the plain package default, which is the shape finding
    27 says to distrust. It is right here, because 'removed' means exactly
    'what the package does now', but it is only right while the package default
    has no rectifier. If one were ever added back, this fails.
    """
    module = load("05_rectifier_removals_at_full_size")
    arms = module.build_arms(seam)
    assert set(arms) == {"kept", "removed"}, f"exp/05 arms changed: {set(arms)}"
    assert differs(arms["kept"], arms["removed"]), (
        f"exp/05's two arms for the '{seam}' seam produce the same output, so "
        f"the experiment is comparing a model with itself"
    )


def test_exp_15_arms_differ_from_the_default_in_what_the_encoder_reads():
    """exp/15 scales what the encoders read; each arm must differ from the default.

    The three expression arms change the composition, so they are compared on
    it with the default built from the same seed. The library arm changes only
    what the library-size encoder reads, so its composition is the default's by
    construction and it is compared on the predicted log library size instead;
    comparing it on the composition would report a false alarm. Finding 27.
    """
    from preffect.config import ModelConfig
    from preffect.model import SimpleModel

    module = load("15_encoder_input_scaling")
    n_genes, n_samples = 20, 8
    arms = module.build_arms(module.example_fit(n_genes))
    expected = {"log1p", "log1p_cpm", "log1p_standardised", "log_pseudocount",
                "library_log", "library_centred"}
    assert set(arms) == expected, set(arms)
    counts = torch.randint(0, 50, (n_samples, n_genes)).float()

    def output(build, infer_library_size: bool):
        config = ModelConfig(hidden_dim=16, latent_dim=4, infer_library_size=infer_library_size)
        torch.manual_seed(0)
        model = build(n_genes, config).eval()
        with torch.no_grad():
            torch.manual_seed(1)
            return model(counts, None)

    reference = output(arms["log1p"].build, False)
    assert isinstance(reference, type(output(lambda n, c: SimpleModel(n, c), False)))
    for name in ("log1p_cpm", "log1p_standardised", "log_pseudocount"):
        other = output(arms[name].build, False)
        assert not torch.allclose(reference.composition, other.composition), (
            f"exp/15's {name} arm produces the default's composition"
        )
    plain = output(arms["library_log"].build, True)
    centred = output(arms["library_centred"].build, True)
    assert torch.allclose(plain.composition, centred.composition), (
        "the library arm should change only what the library encoder reads"
    )
    predicted = (plain.predicted_log_library_size, centred.predicted_log_library_size)
    assert not torch.allclose(*predicted), (
        "exp/15's library_centred arm predicts the same library size as the plain one"
    )


def test_exp_14_arms_are_three_different_masking_schemes():
    """exp/14 compares which entries get hidden, not which model; its arms are schemes.

    On a matrix with zeros, the default scheme may hide entries that are
    already zero, the non-zero schemes never do, and the budget-matched one
    hides the same number per gene as the default. An arm that quietly fell
    back to the default would hide zeros again, which is finding 27's shape
    for a scheme rather than a model.
    """
    import numpy as np

    module = load("14_masking_zeros")
    arms = module.build_arms(0.5)
    assert set(arms) == {"any_entry", "nonzero_only", "nonzero_matched_fraction"}
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 4, size=(20, 6)).astype(np.float32)  # about a quarter zeros
    counts[:, 0] = 0  # one gene with nothing to hide
    masks = {name: build(counts, np.random.default_rng(1)) for name, build in arms.items()}
    hidden = {name: mask.hidden for name, (mask, _) in masks.items()}
    assert (hidden["any_entry"] & (counts == 0)).any(), "the default should hide some zeros here"
    for name in ("nonzero_only", "nonzero_matched_fraction"):
        assert not (hidden[name] & (counts == 0)).any(), f"{name} hid a zero"
    per_gene_default = hidden["any_entry"].sum(axis=0)
    per_gene_matched = hidden["nonzero_only"].sum(axis=0)
    for g in range(6):
        assert per_gene_matched[g] == min(per_gene_default[g], int((counts[:, g] != 0).sum()))
    assert masks["nonzero_only"][1] > 0, "the all-zero gene should be reported as a shortfall"


def test_exp_14_names_its_non_random_schemes_and_corrects_for_them_itself():
    """Finding 56: the library correction is only valid when hiding is random.

    Two of this experiment's three arms hide an entry only if it carries a
    count, so which entries are hidden depends on their values and the
    package's estimate would under-correct. The masks say so, the package
    refuses them, and the experiment supplies arithmetic of its own that gives
    a larger factor, because every entry it hid removed something.
    """
    import numpy as np
    import torch

    from preffect.data.tensors import TensorData
    from preffect.errors import DataError

    module = load("14_masking_zeros")
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 10, size=(30, 12)).astype(np.float32)
    arms = module.build_arms(0.2)

    mcar_mask_, _ = arms["any_entry"](counts, np.random.default_rng(1))
    nonzero_mask, _ = arms["nonzero_only"](counts, np.random.default_rng(1))
    assert mcar_mask_.scheme == "mcar"
    assert nonzero_mask.scheme == "nonzero_only"

    # The package refuses the scheme it cannot estimate for, rather than
    # estimating anyway.
    with pytest.raises(DataError, match="only valid for the 'mcar' scheme"):
        nonzero_mask.library_scale()
    with pytest.raises(DataError, match="only valid for the 'mcar' scheme"):
        TensorData(
            inputs=torch.from_numpy(counts), targets=torch.from_numpy(counts),
            mask=nonzero_mask,
        )

    # The experiment's own arithmetic answers for both, and asks for more
    # where every hidden entry carried a count.
    from_mcar = module.library_scale_for(mcar_mask_, counts)
    from_nonzero = module.library_scale_for(nonzero_mask, counts)
    np.testing.assert_allclose(from_mcar, mcar_mask_.library_scale(), rtol=1e-6)
    assert np.median(from_nonzero) > np.median(from_mcar)

    # And it brings the totals back, which is the thing it is for.
    shown = np.where(nonzero_mask.hidden, 0.0, counts)
    corrected = (shown.sum(axis=1, keepdims=True) * from_nonzero).ravel()
    estimated = corrected / counts.sum(axis=1)
    assert np.abs(np.median(estimated) - 1.0) < 0.05, np.median(estimated)


def test_exp_16_arms_differ_in_the_network_they_read():
    """exp/16 compares networks, not models; its arms are one graph model over different networks.

    The simple arm reads no network, and each graph arm reads one of the
    candidates built from the same correlations. Two arms carrying the same
    network would be finding 27's shape for a network rather than a model, so
    this checks that every candidate is square, symmetric, free of self-links,
    and different from every other, and that the arms carry them unchanged.
    Finding 54 is why the experiment exists: the one network the project
    thought it had was an identity matrix read off its own misalignment.
    """
    import numpy as np

    module = load("16_cohort_sample_network")
    rng = np.random.default_rng(0)
    # Three groups of correlated samples, so the nearest-neighbour candidates
    # have something to find and the threshold one is not complete.
    groups = np.repeat([0, 1, 2], 10)
    signal = rng.normal(size=(3, 40))[groups]
    counts = np.exp(2.0 + signal + 0.6 * rng.normal(size=(30, 40))).astype(np.float32)
    correlation, _used = module.sample_correlations(counts)
    candidates = module.candidate_networks(correlation)
    assert set(candidates) == {"threshold_0.8", "knn_5", "knn_10", "knn_5_mutual"}, set(candidates)

    arms = module.build_arms(candidates)
    assert set(arms) == {"simple", *candidates}
    assert arms["simple"].network is None and arms["simple"].kind == "simple"
    seen = []
    for name, network in candidates.items():
        assert arms[name].kind == "single"
        assert arms[name].network is network, f"{name}'s arm does not carry its candidate"
        assert network.shape == (30, 30) and network.dtype == np.int8
        assert np.array_equal(network, network.T), f"{name} is not symmetric"
        assert not np.diag(network).any(), f"{name} has self-links"
        assert network.any(), f"{name} has no edges"
        assert all(not np.array_equal(network, other) for other in seen), (
            f"{name} repeats another candidate"
        )
        seen.append(network)
    # The mutual network is contained in the union one, and the union of 5 in the union of 10.
    assert not (candidates["knn_5_mutual"] & ~candidates["knn_5"]).any()
    assert not (candidates["knn_5"] & ~candidates["knn_10"]).any()
    assert (candidates["knn_5"].sum(axis=1) >= 5).all()


def test_exp_16_simple_and_graph_arms_build_different_models():
    """The simple arm and a graph arm must not be the same model under two names.

    The graph model is an optional extra, so this skips where it is not
    installed, as `tests/test_graph_model.py` does.
    """
    pytest.importorskip("torch_geometric", reason="needs the graph extra")

    import numpy as np

    from preffect.config import ModelConfig, ModelKind

    module = load("16_cohort_sample_network")
    n_samples, n_genes = 8, 20
    ring = np.eye(n_samples, k=1, dtype=np.int8)
    ring[-1, 0] = 1
    ring = ((ring + ring.T) > 0).astype(np.int8)
    arms = module.build_arms({"ring": ring})
    counts = torch.randint(0, 50, (n_samples, n_genes)).float()

    torch.manual_seed(0)
    simple = arms["simple"].build(n_genes, ModelConfig(hidden_dim=16, latent_dim=4)).eval()
    torch.manual_seed(0)
    graph = arms["ring"].build(
        n_genes, ModelConfig(kind=ModelKind.SINGLE, hidden_dim=16, latent_dim=4, attention_heads=2)
    ).eval()
    with torch.no_grad():
        torch.manual_seed(1)
        a = simple(counts).composition
        torch.manual_seed(1)
        b = graph(counts, None, module.edge_index_of(ring)).composition
    assert not torch.allclose(a, b), "exp/16's simple and graph arms produce the same output"


def test_exp_17_arms_are_three_different_scorings_of_the_network_term():
    """exp/17 compares how the network term is scored, not which model; its arms are terms.

    Two arms are the package's two values of ``edge_balance`` and the third is
    the experiment's own sampled term, which lost and lives there. On a tiny
    network with two edges the three give three different values of the term
    for the same prediction, and the arm named ``none`` gives exactly what the
    term gave before the setting existed. An arm that quietly fell back to
    the default would agree with it, which is finding 27's shape for a setting
    rather than a model.
    """
    import numpy as np

    module = load("17_network_term_weighting")
    arms = module.build_arms()
    assert set(arms) == {"none", "balanced", "sampled"}, set(arms)
    assert arms["none"].weights.edge_balance == "none"
    assert arms["balanced"].weights.edge_balance == "balanced"
    assert arms["sampled"].term is module.sampled_network_term, (
        "the sampled arm must use the experiment's own term; the package has none"
    )

    observed = torch.zeros(6, 6)
    for a, b in [(0, 1), (2, 3)]:
        observed[a, b] = observed[b, a] = 1.0
    torch.manual_seed(0)
    predicted = torch.rand(6, 6) * 0.8 + 0.1
    predicted = (predicted + predicted.t()) / 2
    torch.manual_seed(0)
    values = {name: float(arm.term(predicted, observed)) for name, arm in arms.items()}
    assert len(set(values.values())) == 3, f"two arms score the term the same: {values}"
    before = torch.nn.functional.binary_cross_entropy(predicted.clamp(1e-7, 1 - 1e-7), observed)
    assert values["none"] == float(before)
    # And the random network the experiment pairs with the learnable one has
    # the same number of edges, or the comparison is not at matched density.
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 3, size=60)
    learnable = module.exp08().similarity_network(groups, rng)
    random = module.random_network_like(learnable, np.random.default_rng(1))
    assert int(np.triu(random, 1).sum()) == int(np.triu(learnable, 1).sum())
    assert np.array_equal(random, random.T) and not np.array_equal(random, learnable)


def test_exp_17_sampled_term_scores_every_linked_pair_and_as_many_absent_ones():
    """The experiment's own term, since the package no longer carries it.

    Make every absent pair cost the same, so whichever are drawn the value is
    determined: the mean of the 4 linked entries and 4 absent entries. And the
    draw uses PyTorch's generator, so a seed repeats it and a different seed
    draws different pairs.
    """
    module = load("17_network_term_weighting")
    observed = torch.zeros(6, 6)
    for a, b in [(0, 1), (2, 3)]:
        observed[a, b] = observed[b, a] = 1.0
    predicted = torch.full((6, 6), 0.2)
    predicted[observed > 0] = 0.7
    linked_cost = -torch.log(torch.tensor(0.7))
    absent_cost = -torch.log(torch.tensor(0.8))
    torch.manual_seed(0)
    value = module.sampled_network_term(predicted, observed)
    torch.testing.assert_close(value, (4 * linked_cost + 4 * absent_cost) / 8)

    torch.manual_seed(3)
    varied = torch.rand(6, 6) * 0.8 + 0.1
    varied = (varied + varied.t()) / 2
    torch.manual_seed(11)
    first = module.sampled_network_term(varied, observed)
    torch.manual_seed(11)
    second = module.sampled_network_term(varied, observed)
    torch.manual_seed(12)
    third = module.sampled_network_term(varied, observed)
    assert torch.equal(first, second)
    assert not torch.equal(first, third), "a different seed should draw different absent pairs"


def test_exp_12_arms_are_three_different_stopping_rules():
    """exp/12 compares when to stop, not what to train; its arms are settings.

    The same model under three rules: a fixed length, early stopping on the
    loss, and early stopping on the imputation error (issue #28). An arm that
    quietly lost its rule, or two that shared one, would compare a run with
    itself, which is finding 27's shape for settings rather than models.
    """
    module = load("12_stopping_rule_live")
    arms = module.build_arms(max_epochs=50, fixed_epochs=10)
    assert set(arms) == {"fixed", "stop_on_loss", "stop_on_imputation"}, set(arms)
    assert not arms["fixed"].uses_early_stopping and arms["fixed"].epochs == 10
    assert arms["stop_on_loss"].uses_early_stopping and arms["stop_on_loss"].stop_on == "loss"
    assert arms["stop_on_imputation"].uses_early_stopping
    assert arms["stop_on_imputation"].stop_on == "imputation"
    assert arms["stop_on_loss"].epochs == arms["stop_on_imputation"].epochs == 50


def test_exp_10_single_model_arms_differ():
    """exp/10 asks its question of the graph model too, and those arms must differ.

    The parametrized check above covers `--kind simple`, whose raw arm is a
    subclass of the package default. `--kind single` is the graph model, which
    reads the counts through two encoders of its own, so an arm that stopped
    overriding one of them would not show up there. Finding 27 is about exactly
    this: an arm that quietly becomes the default reports a clean null.

    The graph model is an optional extra, so this skips where it is not
    installed, as `tests/test_graph_model.py` does.
    """
    pytest.importorskip("torch_geometric", reason="needs the graph extra")

    from preffect.config import ModelConfig, ModelKind

    module = load("10_log_counts_at_the_encoder")
    arms = module.build_arms("single")
    assert set(arms) == {"raw", "log1p"}, f"exp/10's single arms changed: {set(arms)}"

    n_genes, n_samples = 20, 8
    config = ModelConfig(
        kind=ModelKind.SINGLE, hidden_dim=16, latent_dim=4, attention_heads=2
    )
    counts = torch.randint(0, 50, (n_samples, n_genes)).float()
    ring = [(i, (i + 1) % n_samples) for i in range(n_samples)]
    edge_index = torch.tensor(ring + [(b, a) for a, b in ring], dtype=torch.long).t()

    produced = []
    for build in (arms["raw"], arms["log1p"]):
        torch.manual_seed(0)
        model = build(n_genes, config).eval()
        with torch.no_grad():
            torch.manual_seed(1)
            produced.append(model(counts, None, edge_index).composition)
    assert not torch.allclose(*produced), (
        "exp/10's 'raw' and 'log1p' arms for the single model produce the same "
        "output, so the experiment is comparing a model with itself"
    )


def builds_more_than_one_model(module) -> bool:
    """Does this experiment compare two models?

    True when it either provides a function that hands back a set of arms, or
    defines its own subclass of one of the package's models. Both are ways of
    saying "this experiment has an arm whose behaviour differs from the
    default", which is the thing that needs checking.
    """
    from preffect import model as package_models

    if callable(getattr(module, "build_arms", None)):
        return True

    shipped = tuple(
        value for value in vars(package_models).values()
        if isinstance(value, type) and issubclass(value, torch.nn.Module)
    )
    return any(
        isinstance(value, type)
        and value not in shipped
        and issubclass(value, shipped)
        for value in vars(module).values()
    )


def test_exp_08_arms_see_different_shares_of_the_network():
    """exp/08 compares one model at two batch sizes.

    What differs between its arms is how much of the sample-sample network an
    epoch can use, so the check that they differ is a check on that share,
    measured the way the run measures it. Equal shares would be finding 27 in
    a new form: two arms trained on the same thing.
    """
    import numpy as np

    module = load("08_graph_model_whole_cohort_batch")
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 3, size=120)
    network = module.similarity_network(groups, rng)
    arms = module.build_arms(epochs=1, n_samples=120)

    shares = {name: module.survival_share(cfg.batch_size, network) for name, cfg in arms.items()}
    assert shares["whole_cohort"] == 1.0
    assert shares["whole_cohort_matched"] == 1.0
    assert shares["mini_batch"] < 0.6, shares
    assert module.within_group_fraction(network, groups) > 0.7
    # The matched arm takes as many optimiser steps as the mini-batch arm.
    steps = -(-120 // arms["mini_batch"].batch_size)
    assert arms["whole_cohort_matched"].epochs == arms["mini_batch"].epochs * steps


def test_every_experiment_with_arms_is_checked_here():
    """Adding an experiment that compares two models must add a test.

    Finding 27 was not only that two arms became the same model. It was that
    nothing in the suite required an experiment to be checked, so the next one
    would have gone the same way unnoticed. This reads `exp/` rather than a
    list, so a new folder cannot slip past by not being added to one.
    """
    for folder in sorted(p for p in EXPERIMENTS.iterdir()
                         if p.is_dir() and p.name[0].isdigit()):
        module = load(folder.name)
        if builds_more_than_one_model(module):
            assert folder.name in ARMS, (
                f"{folder.name} compares more than one model and nothing in "
                f"tests/test_experiments.py checks that its arms differ. Add a "
                f"check and name it in ARMS."
            )
        else:
            assert folder.name in NO_ARMS, (
                f"{folder.name} appears to compare only one model. If that is "
                f"right, say so in NO_ARMS with the reason. If it does compare "
                f"two, this test cannot see how, which is worth fixing."
            )


def test_the_names_this_file_checks_still_exist():
    """ARMS goes stale silently otherwise: a renamed arm would just stop being
    checked, and every test above it would still pass."""
    for experiment, name in ARMS.items():
        module = load(experiment)
        assert getattr(module, name, None) is not None, (
            f"{experiment} no longer defines {name}, so the check for it is "
            f"testing nothing"
        )


def test_every_experiment_folder_has_a_runnable_entry_point():
    """A folder in exp/ that cannot be run is a record of nothing."""
    folders = sorted(
        p for p in EXPERIMENTS.iterdir()
        if p.is_dir() and p.name[0].isdigit()
    )
    assert folders, "no experiments found"
    missing = [p.name for p in folders if not (p / "run_all.py").exists()]
    assert missing == [], f"no run_all.py in: {', '.join(missing)}"


def test_exp_18_arms_are_three_different_models_over_different_tissue_counts():
    """exp/18 compares the three models, so its arms must differ in model and in what they read.

    The three are the simple model with no network, the single model with one,
    and the full model with one per tissue, and the thing that would make this
    experiment report a null it had not earned is two of them turning out to be
    the same model under two names (finding 27). Checked three ways: the kinds
    differ, what each reads differs, and the three built models give three
    different outputs on the same counts. Finding 55 is why the experiment
    exists: the full model's training loop stopped at its first batch, so no
    full-model result existed to compare with anything.
    """
    pytest.importorskip("torch_geometric", reason="needs the graph extra")

    import numpy as np

    from preffect.config import ModelConfig, ModelKind

    module = load("18_full_model_first_run")
    arms = module.build_arms()
    assert set(arms) == {"simple", "single", "full"}, set(arms)
    assert [arms[n].kind for n in ("simple", "single", "full")] == ["simple", "single", "full"]
    assert not arms["simple"].uses_network
    assert arms["single"].uses_network and arms["full"].uses_network
    # The simple and single arms read the target tissue alone; the full arm
    # reads every tissue the data has, which is what `tissues is None` means.
    assert arms["simple"].tissues == 1 and arms["single"].tissues == 1
    assert arms["full"].tissues is None
    assert module.arm_tissue_count(arms["full"], 3) == 3
    assert module.arm_tissue_count(arms["single"], 3) == 1

    n_samples, n_genes = 8, 20
    ring = np.eye(n_samples, k=1, dtype=np.int8)
    ring[-1, 0] = 1
    ring = ((ring + ring.T) > 0).astype(np.int8)
    edges = module.edge_index_of(ring)
    counts = torch.randint(0, 50, (n_samples, n_genes)).float()

    built = {}
    kinds = (("simple", ModelKind.SIMPLE), ("single", ModelKind.SINGLE), ("full", ModelKind.FULL))
    for name, kind in kinds:
        torch.manual_seed(0)
        config = ModelConfig(kind=kind, hidden_dim=16, latent_dim=4, attention_heads=2)
        built[name] = arms[name].build(n_genes, config, 3).eval()

    outputs = {}
    with torch.no_grad():
        torch.manual_seed(1)
        outputs["simple"] = built["simple"](counts).composition
        torch.manual_seed(1)
        outputs["single"] = built["single"](counts, None, edges).composition
        torch.manual_seed(1)
        three = [counts, counts, counts]
        outputs["full"] = built["full"](three, None, [edges, edges, edges]).composition

    for first, second in (("simple", "single"), ("simple", "full"), ("single", "full")):
        assert not torch.allclose(outputs[first], outputs[second]), (
            f"exp/18's {first} and {second} arms produce the same output"
        )


def test_exp_19_arms_differ_in_the_cohort_they_hand_the_model():
    """exp/19 compares what to do with a patient missing a tissue; its arms are cohorts.

    One keeps those patients and pads the tissue they lack, the other drops
    them. Two arms that handed over the same cohort would be finding 27's
    shape for a cohort rather than a model, so this builds one seed's cohort
    and checks that the two differ in exactly the way the experiment claims.
    """
    pytest.importorskip("torch_geometric", reason="needs the graph extra")

    import numpy as np

    from preffect.data.masking import mcar_mask

    module = load("19_unmatched_samples_padded")
    arms = module.build_arms()
    assert set(arms) == {"padded", "dropped"}, set(arms)
    assert arms["padded"].unmatched == "pad" and arms["padded"].keeps_incomplete_patients
    assert arms["dropped"].unmatched == "refuse"
    assert not arms["dropped"].keeps_incomplete_patients

    n_patients, n_genes = 60, 20
    cohort = module.synthetic_cohort(0, n_patients, n_genes, 0.2)
    assert int((~cohort.complete).sum()) == 12
    assert [t.n_obs for t in cohort.tissues] == [60, 48]
    mask = mcar_mask(n_patients, n_genes, 0.2, np.random.default_rng(1))

    padded, held_padded = module.cohort_for_arm(cohort, arms["padded"], mask)
    dropped, held_dropped = module.cohort_for_arm(cohort, arms["dropped"], mask)

    assert padded.n_samples == n_patients
    assert dropped.n_samples == n_patients - 12
    assert padded.n_ghosts == [0, 12], padded.n_ghosts
    assert dropped.n_ghosts == [0, 0], dropped.n_ghosts
    assert held_padded.all()
    assert np.array_equal(held_dropped, cohort.complete)
    # The predicted tissue is complete in both arms, so neither leaves rows out
    # of the expression term; that rule is checked in tests/test_full_model.py.
    assert padded.target_real_rows() is None
    assert dropped.target_real_rows() is None
