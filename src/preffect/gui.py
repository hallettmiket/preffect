"""The graphical interface: check a folder of counts, apply a saved model, look.

Started with ``preffect gui``, which runs this file under Streamlit and opens
it in a browser. It runs on this machine only and sends nothing anywhere;
Streamlit's own usage reporting is switched off by the command that starts it.

It covers the part of the work a person wants to *see*: whether a folder of
counts is usable, what a fitted model makes of it, and the figures that say
whether to believe it. Fitting a model is hours on a real cohort and belongs
on the command line, where it can be left running; the interface reads what
``preffect train`` wrote. Every figure here is one of ``preffect.plots``'
figures, drawn by the same call a script would make, so a picture seen here
can be reproduced and put in a paper.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

__all__ = ["render"]


def _as_dense(values):
    """Counts as a dense array, whatever the file stored them as."""
    import numpy as np

    return np.asarray(values.toarray() if hasattr(values, "toarray") else values)


def _colour_columns(obs, limit: int = 20) -> list[str]:
    """Columns of the sample table that look like labels: few distinct values."""
    return [c for c in obs.columns if obs[c].nunique(dropna=False) <= limit]


def render() -> None:
    """Draw the page. Streamlit runs this again on every interaction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import streamlit as st

    from preffect import __version__, plots
    from preffect.errors import PreffectError

    st.set_page_config(page_title="PREFFECT", layout="wide")
    st.title(f"PREFFECT {__version__}")
    st.caption("Runs on this machine only. Nothing you point it at leaves it.")

    with st.sidebar:
        st.header("Data")
        data_folder = st.text_input(
            "Folder of counts",
            key="data_folder",
            help="One file per tissue, named <anything>tau_<n>.h5ad.",
        )
        n_tissues = int(
            st.number_input("Tissue files in the folder", min_value=1, value=1, key="n_tissues")
        )
        check = st.button("Check the data", key="check")

        st.header("Model")
        model_path = st.text_input(
            "Saved model (.pt)", key="model_path", help="What preffect train wrote."
        )
        device = st.selectbox("Run on", ["cpu", "cuda"], key="device")
        apply_ = st.button("Apply the model", key="apply")

    # ----- check: what the command line's `check` says, as a table
    if check:
        from preffect.workflow import inspect_split

        if not data_folder:
            st.error("Give a folder of counts first.")
        else:
            st.session_state["report"] = inspect_split(Path(data_folder), n_tissues)

    report = st.session_state.get("report")
    if report is not None:
        st.header("The data")
        st.caption(str(report.folder))
        if report.tissues:
            st.table(
                [
                    {"tissue": i, "samples": t.samples, "genes": t.genes, "network": t.network}
                    for i, t in enumerate(report.tissues, start=1)
                ]
            )
        if report.usable:
            st.success("The counts look usable.")
        else:
            st.error(report.problem)

    # ----- apply: the command line's `apply`, kept in memory for the figures
    if apply_:
        if not data_folder or not model_path:
            st.error("Give both a folder of counts and a saved model.")
        else:
            from preffect.config import RuntimeConfig
            from preffect.training import resolve_device
            from preffect.workflow import apply_checkpoint

            try:
                with st.spinner("Applying the model"):
                    resolved = resolve_device(RuntimeConfig(device=device))
                    st.session_state["applied"] = apply_checkpoint(
                        Path(model_path), Path(data_folder), n_tissues=n_tissues, device=resolved
                    )
            except PreffectError as error:
                st.error(str(error))

    applied = st.session_state.get("applied")
    if applied is None:
        return
    result, tissues, checkpoint = applied
    target = tissues[0]
    observed = _as_dense(target.X)
    gene_names = [str(n) for n in target.var_names]

    st.header("What the model made of it")
    columns = st.columns(4)
    columns[0].metric("model", checkpoint.model_config.kind.value)
    # Which count distribution the model was fitted with. Shown rather than
    # chosen: this page applies a saved model and does not fit one, so the
    # choice belongs to `preffect train --likelihood`. A ZINB model and an NB
    # model of the same kind are otherwise indistinguishable here, and the
    # difference changes what the predicted counts mean.
    columns[1].metric("likelihood", checkpoint.model_config.likelihood.value)
    columns[2].metric("samples", target.n_obs)
    columns[3].metric("genes matched", target.n_vars)

    # The same file `preffect apply` writes, offered for download.
    adata = result.to_anndata(
        sample_names=[str(n) for n in target.obs_names], gene_names=gene_names
    )
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "cleaned.h5ad"
        adata.write_h5ad(path)
        payload = path.read_bytes()
    st.download_button(
        "Download the cleaned counts (.h5ad)",
        data=payload,
        file_name="cleaned.h5ad",
        help="Cleaned counts in X, the predicted proportions in layers['composition'], "
        "the latent summary in obsm['latent'].",
    )

    st.subheader("Before and after")
    figures = [
        (
            "Library sizes: observed against inferred",
            lambda: plots.library_size_histogram(result, observed),
        ),
        (
            "Observed against fitted counts, the most variable genes",
            lambda: plots.gene_scatter(result, observed, gene_names=gene_names),
        ),
        (
            "Gene means, dispersion and library size",
            lambda: plots.library_size_and_dispersion(result, observed),
        ),
    ]
    if result.zero_logits is not None:
        figures.append(
            (
                "How often a gene is zero, against the model's own zero probability",
                lambda: plots.zero_fraction_against_pi(result, observed),
            )
        )
    for title, draw in figures:
        st.markdown(f"**{title}**")
        figure = draw()
        st.pyplot(figure)
        plt.close(figure)

    st.subheader("The latent summary")
    choices = ["none"] + _colour_columns(target.obs)
    colour = st.selectbox("Colour by", choices, key="colour_by")
    colour_by = None if colour == "none" else target.obs[colour].astype(str).to_numpy()
    try:
        figure = plots.latent_umap(result, colour_by=colour_by, colour_label=colour)
    except ImportError:
        st.info('The latent figure needs the viz extra: pip install "preffect[viz]".')
    else:
        st.pyplot(figure)
        plt.close(figure)


if __name__ == "__main__":
    render()
