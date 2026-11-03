"""SHAFLB dashboard: `streamlit run app.py`."""

import json
import re

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src import charts
from src.blockchain import AuditLedger, consensus
from src.config import (
    ARRAYS_FILE,
    EXCLUDED_DISCHARGE_IDS,
    HOSPITAL_IDS,
    HOSPITALS,
    LEDGER_FILE,
    MODEL_BUNDLE_FILE,
    RAW_DATA_FILE,
    FLConfig,
)
from src.data import download_dataset, load_id_labels
from src.federated import partition_non_iid, predict_proba, run_federated
from src.models import METRIC_LABELS, compute_metrics
from src.pipeline import artifacts_exist, load_results, run_pipeline

st.set_page_config(
    page_title="SHAFLB · Federated Hospital Analytics",
    page_icon=":material/shield_with_heart:",
    layout="wide",
)

st.html(
    """
    <style>
      .block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1280px; }
      .kicker { font-size: .75rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; opacity: .6; margin-bottom: .1rem; }
      .lead { font-size: 1.05rem; opacity: .78; max-width: 62rem; line-height: 1.55; margin-bottom: .6rem; }
      .step-num { font-size: .75rem; font-weight: 700; opacity: .5; letter-spacing: .06em; }
      [data-testid="stMetricValue"] { font-weight: 650; }
    </style>
    """
)

LOGO_FILE = Path(__file__).parent / "assets" / "logo.svg"
ICON_FILE = Path(__file__).parent / "assets" / "icon.svg"
MODEL_KEYS = ["centralized_lr", "centralized_rf", "fedavg", "fedprox", "dp_fedavg"]
MAIN_METRICS = ["roc_auc", "pr_auc", "f1", "recall", "precision", "specificity", "accuracy"]


# ── data access ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_results() -> dict:
    return load_results()


@st.cache_data(show_spinner=False)
def get_arrays() -> dict:
    with np.load(ARRAYS_FILE) as data:
        return {k: data[k] for k in data.files}


@st.cache_resource(show_spinner=False)
def get_bundle() -> dict:
    return joblib.load(MODEL_BUNDLE_FILE)


@st.cache_data(show_spinner=False)
def get_id_labels() -> dict:
    return load_id_labels()


@st.cache_resource(show_spinner=False)
def get_clients():
    arrays = get_arrays()
    return partition_non_iid(arrays["X_train"], arrays["y_train"])


def clear_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()
    for key in ("replicas", "lab_run"):
        st.session_state.pop(key, None)


def run_experiment_with_progress() -> None:
    bar = st.progress(0.0, text="Starting")
    run_pipeline(progress=lambda frac, msg: bar.progress(min(frac, 1.0), text=msg))
    clear_caches()
    st.rerun()


# ── helpers ──────────────────────────────────────────────────────────────────
def header(kicker: str, title: str, lead: str) -> None:
    st.html(f'<div class="kicker">{kicker}</div>')
    st.title(title, anchor=False)
    st.html(f'<div class="lead">{lead}</div>')


def model_label(key: str) -> str:
    return charts.MODEL_NAMES[key]


NUMERIC_NAMES = {
    "time_in_hospital": "Length of stay (days)",
    "num_lab_procedures": "Lab procedures",
    "num_procedures": "Procedures",
    "num_medications": "Medications",
    "number_outpatient": "Prior outpatient visits",
    "number_emergency": "Prior emergency visits",
    "number_inpatient": "Prior inpatient admissions",
    "number_diagnoses": "Number of diagnoses",
}
CATEGORY_NAMES = {
    "race": "Race", "gender": "Gender", "age": "Age", "admission_type_id": "Admission",
    "discharge_disposition_id": "Discharge", "admission_source_id": "Source",
    "max_glu_serum": "Glucose serum", "A1Cresult": "HbA1c", "insulin": "Insulin",
    "change": "Medication change", "diabetesMed": "Diabetes medication",
    "diag_1_group": "Primary dx", "diag_2_group": "Secondary dx", "diag_3_group": "Tertiary dx",
}


def pretty_feature(name: str) -> str:
    if name in NUMERIC_NAMES:
        return NUMERIC_NAMES[name]
    labels = get_id_labels()
    for col in sorted(CATEGORY_NAMES, key=len, reverse=True):
        if name.startswith(col + "_"):
            value = name[len(col) + 1:]
            if col in labels:
                value = labels[col].get(value, value)
            value = re.sub(r"^\[(\d+)-(\d+)\)$", r"\1–\2", value)
            return f"{CATEGORY_NAMES[col]}: {value[:38]}"
    col, _, value = name.rpartition("_")
    return f"{col.capitalize()}: {value}"


# ── pages ────────────────────────────────────────────────────────────────────
def page_overview() -> None:
    r = get_results()
    m, ds = r["models"], r["dataset"]
    lr, fed = m["centralized_lr"], m["fedavg"]
    header(
        "Federated learning · Audit ledger · Clinical risk",
        "Secure Hospital Analytics",
        "Five hospitals jointly train a 30-day readmission model for diabetic patients "
        "without sharing a single patient record. Only model parameters move; every training "
        "round is written to a tamper-evident, hash-chained ledger replicated across the network.",
    )

    c = st.columns(4)
    c[0].metric("Patients in cohort", f"{ds['total_records']:,}", border=True,
                help="Unique patients after excluding hospice / expired discharges")
    c[1].metric("Readmitted < 30 days", f"{ds['readmission_rate_percent']}%", border=True,
                help=f"{ds['positive_samples']:,} positive cases: heavy class imbalance")
    c[2].metric("Federated ROC-AUC", f"{fed['roc_auc']:.3f}", f"{fed['roc_auc'] - lr['roc_auc']:+.3f} vs centralised",
                delta_color="off", border=True)
    c[3].metric("Ledger blocks verified", f"{r['ledger']['blocks']}", "chain intact" if r["ledger"]["verified"] else "BROKEN",
                delta_color="normal" if r["ledger"]["verified"] else "inverse", border=True)

    retained = 100 * fed["roc_auc"] / lr["roc_auc"]
    gains = [fed["roc_auc"] - h["metrics"]["roc_auc"] for h in r["local_only"]]
    findings = [
        ("Privacy at almost no cost",
         f"FedAvg reaches ROC-AUC **{fed['roc_auc']:.3f}** versus **{lr['roc_auc']:.3f}** for a model trained on "
         f"pooled data, retaining **{retained:.1f}%** of centralised performance."),
        ("Collaboration pays",
         f"Every hospital does better inside the federation: ROC-AUC gains of **+{min(gains):.3f}** to "
         f"**+{max(gains):.3f}** over a model trained on its own data alone."),
        ("Differential privacy has a price",
         f"Clipping and noising updates (DP-FedAvg) lowers ROC-AUC to **{m['dp_fedavg']['roc_auc']:.3f}**, "
         "the privacy/utility trade-off a real deployment must tune."),
        ("Tampering is caught",
         "Editing any recorded round breaks the hash chain, and a replica rewritten to hide the edit is "
         "outvoted by the other hospitals' copies."),
    ]
    st.subheader("Key findings", anchor=False)
    for row in (findings[:2], findings[2:]):
        for col, (title, text) in zip(st.columns(2), row):
            with col.container(border=True, height="stretch"):
                st.markdown(f"**{title}**  \n{text}")

    st.subheader("How it works", anchor=False)
    with st.container(border=True):
        st.graphviz_chart(ARCHITECTURE_DOT, width="stretch")
    steps = [
        ("Broadcast", "The coordinator sends the current global model to every hospital."),
        ("Train locally", "Each hospital runs class-weighted SGD on its private records. Data never leaves."),
        ("Aggregate", "The server averages the returned parameters, weighted by each hospital's sample count."),
        ("Record", "Round metadata and a SHA-256 fingerprint of the new model are appended to the ledger."),
    ]
    for col, (i, (title, text)) in zip(st.columns(4), enumerate(steps, 1)):
        with col.container(border=True, height="stretch"):
            st.html(f'<div class="step-num">STEP {i}</div>')
            st.markdown(f"**{title}**  \n{text}")
    st.page_link(PAGES["about"], label="Read the full project description", icon=":material/arrow_forward:")


ARCHITECTURE_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent"; nodesep=0.3; ranksep=1.1; size="11,2.2";
  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=13 penwidth=0 margin="0.2,0.1"];
  edge [color="#8a8a85" arrowsize=0.7 fontname="Helvetica" fontsize=11 fontcolor="#8a8a85"];
  H [label="5 hospitals\\nprivate records\\nlocal SGD" fillcolor="#cde2fb" fontcolor="#0d366b"];
  S [label="Aggregation\\nserver\\nFedAvg / FedProx" fillcolor="#fde0d2" fontcolor="#7a2a0b"];
  M [label="Global\\nmodel" fillcolor="#fde0d2" fontcolor="#7a2a0b"];
  L [label="Audit ledger\\nSHA-256 chain\\n5 replicas" fillcolor="#d3f1e5" fontcolor="#0b4f37"];
  H -> S [label="weights only"];
  S -> M;
  M -> L [label="fingerprint"];
  M -> H [label="broadcast" style=dashed constraint=false];
}
"""


def page_hospitals() -> None:
    r = get_results()
    header(
        "Data",
        "Hospital network",
        "The training split is divided across five simulated hospitals with deliberately different "
        "sizes and readmission rates (non-IID), as happens between real institutions.",
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Cohort size by hospital", anchor=False)
        st.plotly_chart(charts.hospital_volumes(r["hospitals"]))
    with right:
        st.subheader("Readmission rate by hospital", anchor=False)
        st.plotly_chart(charts.readmission_rates(r["hospitals"], r["dataset"]["readmission_rate_percent"]))

    df = pd.DataFrame(r["hospitals"])
    df["hospital_id"] = df["hospital_id"].str.replace("_", " ")
    st.dataframe(
        df[["hospital_id", "profile", "total_samples", "positive_samples", "readmission_rate_pct"]],
        hide_index=True,
        column_config={
            "hospital_id": "Hospital",
            "profile": "Profile",
            "total_samples": st.column_config.NumberColumn("Patients", format="localized"),
            "positive_samples": st.column_config.NumberColumn("Readmitted", format="localized"),
            "readmission_rate_pct": st.column_config.ProgressColumn(
                "Readmission rate", min_value=0, max_value=15, format="%.2f%%"),
        },
    )
    ds, sp = r["dataset"], r["split"]
    st.caption(
        f"Source: UCI Diabetes 130-US Hospitals (1999–2008). {ds['total_records']:,} patients after cohort "
        f"selection; stratified split into {sp['train']:,} training and {sp['test']:,} held-out test patients; "
        f"{sp['features']} features after one-hot encoding. The test set stays with the evaluator and is never partitioned."
    )


def page_performance() -> None:
    r, arrays = get_results(), get_arrays()
    models = r["models"]
    header(
        "Evaluation",
        "Model performance",
        "All models are scored on the same held-out test set. With only 9% positives, accuracy is misleading; "
        "focus on ROC-AUC, PR-AUC, recall and F1.",
    )

    chosen = st.pills(
        "Models", MODEL_KEYS, selection_mode="multi", default=["centralized_lr", "fedavg", "fedprox"],
        format_func=model_label,
    ) or ["fedavg"]

    rows = [{"Model": model_label(k), **{METRIC_LABELS[m]: models[k][m] for m in MAIN_METRICS}} for k in chosen]
    st.dataframe(
        pd.DataFrame(rows), hide_index=True,
        column_config={METRIC_LABELS[m]: st.column_config.NumberColumn(format="%.4f") for m in MAIN_METRICS},
    )

    tab_roc, tab_pr, tab_bars, tab_cm, tab_thr = st.tabs(
        ["ROC curve", "Precision–recall", "Metric comparison", "Confusion matrices", "Threshold explorer"])
    y_test = arrays["y_test"]
    probas = {k: arrays[f"proba_{k}"] for k in chosen}
    with tab_roc:
        st.plotly_chart(charts.roc_pr_curves(y_test, probas, "roc"), key="roc")
    with tab_pr:
        st.plotly_chart(charts.roc_pr_curves(y_test, probas, "pr"), key="pr")
    with tab_bars:
        st.plotly_chart(charts.metric_bars(models, chosen, MAIN_METRICS[:-1]), key="bars")
    with tab_cm:
        for col, k in zip(st.columns(min(len(chosen), 3)), chosen[:3]):
            with col:
                st.plotly_chart(charts.confusion(models[k]["confusion_matrix"], model_label(k)), key=f"cm_{k}")
        if len(chosen) > 3:
            st.caption("Showing the first three selected models.")
    with tab_thr:
        st.markdown(
            "Models are trained with class weighting, so scores are **risk rankings, not calibrated probabilities**. "
            "Moving the threshold trades missed readmissions (recall) against false alarms (precision)."
        )
        thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
        for k in chosen:
            mt = compute_metrics(y_test, probas[k], thr)
            flagged = (probas[k] >= thr).mean()
            st.markdown(f"**{model_label(k)}**")
            c = st.columns(5)
            c[0].metric("Recall", f"{mt['recall']:.1%}", border=True)
            c[1].metric("Precision", f"{mt['precision']:.1%}", border=True)
            c[2].metric("F1", f"{mt['f1']:.3f}", border=True)
            c[3].metric("Specificity", f"{mt['specificity']:.1%}", border=True)
            c[4].metric("Patients flagged", f"{flagged:.1%}", border=True)


def page_federated() -> None:
    r = get_results()
    runs, models = r["fl_runs"], r["models"]
    header(
        "Training",
        "Federated training",
        "How the global model evolves across communication rounds, how the aggregation strategies compare, "
        "and what each hospital gains by collaborating instead of training alone.",
    )
    metric = st.segmented_control(
        "Metric", ["roc_auc", "pr_auc", "f1", "recall"], default="roc_auc",
        format_func=lambda k: METRIC_LABELS[k], required=True,
    )
    st.subheader("Convergence across rounds", anchor=False)
    st.plotly_chart(charts.convergence(runs, metric, reference=models["centralized_lr"]))

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader("Value of collaboration", anchor=False)
        st.caption("Each bar is a model trained on that hospital's data only, scored on the shared test set.")
        st.plotly_chart(charts.local_vs_federated(r["local_only"], models["fedavg"], metric))
    with right:
        st.subheader("Aggregation strategies", anchor=False)
        rows = []
        for key, run in runs.items():
            cfg = run["config"]
            setting = {
                "fedavg": "Sample-weighted averaging",
                "fedprox": f"Proximal term μ = {cfg['prox_mu']}",
                "dp_fedavg": f"Clip {cfg['clip_norm']}, noise σ = {cfg['noise_multiplier']}",
            }[key]
            rows.append({"Strategy": run["label"], "Setting": setting,
                         "ROC-AUC": models[key]["roc_auc"], "F1": models[key]["f1"], "Recall": models[key]["recall"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     column_config={k: st.column_config.NumberColumn(format="%.4f") for k in ("ROC-AUC", "F1", "Recall")})
        st.markdown(
            "- **FedAvg** averages local models weighted by sample count.\n"
            "- **FedProx** adds a proximal penalty that keeps local models close to the global one, "
            "which stabilises training on non-IID data.\n"
            "- **DP-FedAvg** clips each hospital's update and adds Gaussian noise to the aggregate, "
            "limiting what any single hospital's contribution reveals."
        )

    st.subheader("Client update magnitudes (FedAvg)", anchor=False)
    st.caption("L2 norm of each hospital's parameter update per round. The server can monitor this signal to flag anomalous "
               "or poisoned contributions; it is also recorded alongside each round.")
    st.plotly_chart(charts.update_norms(runs["fedavg"]["history"]))

    with st.expander("Round-by-round detail"):
        run_key = st.radio("Run", list(runs), format_func=lambda k: runs[k]["label"], horizontal=True)
        st.dataframe(
            pd.DataFrame([{"Round": h["round"], **{METRIC_LABELS[m]: h["metrics"][m] for m in MAIN_METRICS}}
                          for h in runs[run_key]["history"]]),
            hide_index=True,
        )


def page_lab() -> None:
    header(
        "Interactive",
        "Federated learning lab",
        "Configure and run a fresh federated training job on the real hospital partitions. "
        "Try dropping hospitals, adding differential-privacy noise, or changing local work per round.",
    )
    r = get_results()
    with st.container(border=True):
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            algorithm = st.segmented_control("Algorithm", ["fedavg", "fedprox"], default="fedavg", required=True,
                                             format_func=lambda a: {"fedavg": "FedAvg", "fedprox": "FedProx"}[a])
            mu = st.slider("Proximal μ", 0.0, 1.0, 0.05, 0.01, disabled=algorithm != "fedprox",
                           help="Strength of the pull towards the global model (FedProx only)")
            hospitals = st.multiselect("Participating hospitals", HOSPITAL_IDS, default=HOSPITAL_IDS,
                                       format_func=lambda h: h.replace("_", " "))
        with c2:
            rounds = st.slider("Communication rounds", 1, 20, 8)
            epochs = st.slider("Local epochs per round", 1, 5, 3)
            lr = st.select_slider("Learning rate", [0.005, 0.01, 0.02, 0.05, 0.1, 0.2], value=0.05)
            batch = st.select_slider("Batch size", [16, 32, 64, 128, 256], value=64)
        with c3:
            dp = st.toggle("Differential privacy", help="Clip each hospital's update and add Gaussian noise")
            clip = st.slider("Clip norm", 0.1, 3.0, 1.0, 0.1, disabled=not dp)
            noise = st.slider("Noise multiplier σ", 0.0, 1.0, 0.1, 0.05, disabled=not dp)
            st.space("small")
            start = st.button("Run training", type="primary", icon=":material/play_arrow:", width="stretch",
                           disabled=not hospitals)

    if start:
        cfg = FLConfig(
            rounds=rounds, local_epochs=epochs, learning_rate=lr, batch_size=batch, algorithm=algorithm,
            prox_mu=mu if algorithm == "fedprox" else 0.0,
            clip_norm=clip if dp else 0.0, noise_multiplier=noise if dp else 0.0,
        )
        arrays = get_arrays()
        clients = [c for c in get_clients() if c.hospital_id in hospitals]
        ledger = AuditLedger(run_config={"algorithm": cfg.label, "hospitals": hospitals, **cfg.to_dict()})
        bar = st.progress(0.0, text="Training")

        def on_round(rec, w, b):
            ledger.add_round(rec, w, b)
            bar.progress(rec["round"] / cfg.rounds,
                         text=f"Round {rec['round']}/{cfg.rounds} · ROC-AUC {rec['metrics']['roc_auc']:.4f}")

        run = run_federated(clients, arrays["X_test"], arrays["y_test"], cfg, on_round=on_round)
        bar.empty()
        st.session_state.lab_run = {"run": run, "label": cfg.label, "ledger": ledger, "hospitals": hospitals}

    lab = st.session_state.get("lab_run")
    if not lab:
        st.info("Set the parameters above and press **Run training**. A default run takes about a second.",
                icon=":material/science:")
        return

    final, ref = lab["run"]["history"][-1]["metrics"], r["models"]["centralized_lr"]
    n_h = len(lab["hospitals"])
    st.subheader(f"Result: {lab['label']} across {n_h} hospital{'s' if n_h > 1 else ''}", anchor=False)
    c = st.columns(4)
    for col, key in zip(c, ["roc_auc", "pr_auc", "f1", "recall"]):
        col.metric(METRIC_LABELS[key], f"{final[key]:.4f}", f"{final[key] - ref[key]:+.4f} vs centralised",
                   delta_color="off", border=True)
    runs = {"fedavg": {"label": "This run", "history": lab["run"]["history"]}}
    st.plotly_chart(charts.convergence(runs, "roc_auc", reference=ref), key="lab_conv")
    valid, msg, _ = lab["ledger"].verify()
    (st.success if valid else st.error)(
        f"This run was logged to its own ledger: {msg} Head hash `{lab['ledger'].head[:16]}…`",
        icon=":material/verified:" if valid else ":material/error:")


def page_ledger() -> None:
    header(
        "Audit",
        "Audit ledger",
        "Each FedAvg round is a block holding contributions, metrics and a SHA-256 fingerprint of the model, "
        "linked to the previous block's hash. Every hospital keeps a replica. No patient data is stored.",
    )
    if "replicas" not in st.session_state:
        base = AuditLedger.load(LEDGER_FILE)
        st.session_state.replicas = {h: base.copy() for h in HOSPITAL_IDS}
    replicas: dict[str, AuditLedger] = st.session_state.replicas
    result = consensus(replicas)

    st.subheader("Network status", anchor=False)
    cols = st.columns(len(replicas))
    for col, (node, s) in zip(cols, result["status"].items()):
        ok = s["valid"] and s["agrees"]
        with col.container(border=True):
            st.markdown(f"**{node.replace('_', ' ')}**")
            if ok:
                st.badge("In consensus", icon=":material/check_circle:", color="green")
            elif not s["valid"]:
                st.badge(f"Broken at #{s['bad_index']}", icon=":material/error:", color="red")
            else:
                st.badge("Outvoted", icon=":material/warning:", color="orange")
            st.caption(f"head `{replicas[node].head[:10]}…`")
    if result["quorum"]:
        st.caption(f"Consensus head `{result['majority_head'][:24]}…` held by {result['votes']} of "
                   f"{result['nodes']} replicas. Only replicas that pass their own hash check may vote.")
    else:
        st.error("No majority: the network cannot agree on the ledger state.", icon=":material/error:")

    with st.container(border=True):
        st.markdown("**Tamper lab** · simulate an insider editing a hospital's copy of the ledger")
        c1, c2, c3, c4 = st.columns([1, 0.8, 1, 1])
        node = c1.selectbox("Replica", HOSPITAL_IDS, index=2, format_func=lambda h: h.replace("_", " "))
        n_blocks = len(replicas[node])
        block_idx = c2.selectbox("Block", list(range(1, n_blocks)), index=min(1, n_blocks - 2))
        field = c3.selectbox("Field", ["total_samples", "f1", "roc_auc", "model_hash"],
                             format_func={"total_samples": "Sample count", "f1": "F1 metric",
                                          "roc_auc": "ROC-AUC metric", "model_hash": "Model fingerprint"}.get)
        cover = c4.toggle("Cover tracks", help="The attacker rewrites every later block so the local chain verifies again")
        b1, b2, _ = st.columns([1, 1, 2])
        if b1.button("Apply attack", icon=":material/bug_report:", width="stretch"):
            forged = {"total_samples": 999_999, "f1": 0.99, "roc_auc": 0.99, "model_hash": "f" * 64}[field]
            replicas[node].tamper(block_idx, field, forged)
            if cover:
                replicas[node].rehash_from(block_idx)
            st.rerun()
        if b2.button("Restore all replicas", icon=":material/restart_alt:", width="stretch"):
            st.session_state.pop("replicas")
            st.rerun()
        st.caption("Without covering tracks the edited replica fails its own hash check. With covering tracks it verifies "
                   "locally, but its head hash no longer matches the other hospitals, so it is outvoted.")

    st.subheader("Block explorer", anchor=False)
    flagged = [n for n, s in result["status"].items() if not (s["valid"] and s["agrees"])]
    view = st.selectbox("Replica to inspect", HOSPITAL_IDS, index=HOSPITAL_IDS.index(flagged[0]) if flagged else 0,
                        format_func=lambda h: h.replace("_", " "))
    ledger = replicas[view]
    _, _, bad = ledger.verify()
    rows = []
    for b in ledger.chain:
        rows.append({
            "Block": b.index,
            "Round": b.fl_round or "genesis",
            "Status": "tampered" if bad is not None and b.index == bad else ("unverified" if bad is not None and b.index > bad else "valid"),
            "Samples": b.total_samples,
            "F1": b.metrics.get("f1"),
            "ROC-AUC": b.metrics.get("roc_auc"),
            "Model fingerprint": b.model_hash[:16] + "…",
            "Previous hash": b.previous_hash[:16] + "…",
            "Block hash": b.block_hash[:16] + "…",
            "Timestamp": b.timestamp,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    c1, c2 = st.columns([3, 1])
    with c1.expander("Raw block JSON"):
        idx = st.number_input("Block index", 0, len(ledger) - 1, 0)
        st.json(ledger.to_list()[int(idx)])
    c2.download_button("Download ledger", json.dumps(ledger.to_list(), indent=2), "audit_chain.json",
                       "application/json", icon=":material/download:", width="stretch")


def page_simulator() -> None:
    header(
        "Decision support demo",
        "Readmission risk simulator",
        "Describe a patient encounter to score it with both the centralised and the federated model, "
        "and see which factors drive the federated model's score.",
    )
    bundle, labels = get_bundle(), get_id_labels()
    pre = bundle["preprocessor"]
    cats = dict(zip(pre.named_transformers_["cat"].feature_names_in_, pre.named_transformers_["cat"].categories_))

    def pick(label, col, default=None, exclude=()):
        options = [o for o in cats[col] if o not in {str(e) for e in exclude} and o not in ("Missing", "?")]
        names = labels.get(col, {})
        fmt = (lambda o: f"{names.get(o, o)}") if names else (lambda o: o)
        return st.selectbox(label, options, index=options.index(default) if default in options else 0, format_func=fmt)

    patient = {}
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1.container(border=True, height="stretch"):
        st.markdown("**Patient**")
        patient["age"] = pick("Age band", "age", "[70-80)")
        patient["gender"] = pick("Gender", "gender", "Female")
        patient["race"] = pick("Race", "race", "Caucasian")
        patient["number_inpatient"] = st.slider("Inpatient admissions, past year", 0, 20, 0)
        patient["number_emergency"] = st.slider("Emergency visits, past year", 0, 20, 0)
        patient["number_outpatient"] = st.slider("Outpatient visits, past year", 0, 20, 0)
    with c2.container(border=True, height="stretch"):
        st.markdown("**This admission**")
        patient["admission_type_id"] = pick("Admission type", "admission_type_id", "1")
        patient["admission_source_id"] = pick("Admission source", "admission_source_id", "7")
        patient["discharge_disposition_id"] = pick("Discharge disposition", "discharge_disposition_id", "1",
                                                   exclude=EXCLUDED_DISCHARGE_IDS)
        patient["time_in_hospital"] = st.slider("Length of stay (days)", 1, 14, 4)
        patient["diag_1_group"] = pick("Primary diagnosis", "diag_1_group", "Circulatory")
        patient["diag_2_group"] = pick("Secondary diagnosis", "diag_2_group", "Diabetes")
        patient["diag_3_group"] = pick("Tertiary diagnosis", "diag_3_group", "Circulatory")
    with c3.container(border=True, height="stretch"):
        st.markdown("**Clinical detail**")
        patient["number_diagnoses"] = st.slider("Number of diagnoses", 1, 16, 8)
        patient["num_lab_procedures"] = st.slider("Lab procedures", 1, 130, 44)
        patient["num_procedures"] = st.slider("Procedures", 0, 6, 1)
        patient["num_medications"] = st.slider("Distinct medications", 1, 80, 16)
        patient["A1Cresult"] = pick("HbA1c result", "A1Cresult", "None")
        patient["max_glu_serum"] = pick("Max glucose serum", "max_glu_serum", "None")
        patient["insulin"] = pick("Insulin", "insulin", "Steady")
        patient["change"] = pick("Diabetes medication changed", "change", "No")
        patient["diabetesMed"] = pick("On diabetes medication", "diabetesMed", "Yes")
    for col, values in cats.items():  # remaining medications default to "No"
        patient.setdefault(col, "No" if "No" in values else values[0])

    x = pre.transform(pd.DataFrame([patient]))
    fed = bundle["federated"]
    p_central = float(bundle["centralized_lr"].predict_proba(x)[0, 1])
    p_fed = float(predict_proba(x, fed["weights"], fed["bias"])[0])

    st.subheader("Risk scores", anchor=False)
    g1, g2, g3 = st.columns([1, 1, 1.3], gap="large")
    for col, name, p, key in ((g1, "Centralised model", p_central, "centralized_lr"), (g2, "Federated model", p_fed, "fedavg")):
        with col.container(border=True):
            st.markdown(f"**{name}**")
            st.plotly_chart(charts.risk_gauge(p, charts.MODEL_COLORS[key]), key=f"gauge_{key}")
            if p >= 0.5:
                st.badge("Flag for follow-up", icon=":material/priority_high:", color="red")
            else:
                st.badge("Routine discharge", icon=":material/check:", color="green")
    with g3.container(border=True, height="stretch"):
        st.markdown("**Suggested action**")
        if max(p_central, p_fed) >= 0.5:
            st.markdown("- Follow-up call within 48 hours\n- Medication reconciliation before discharge\n"
                        "- Outpatient appointment within 7–10 days")
        else:
            st.markdown("- Standard discharge instructions\n- Routine primary-care follow-up")
        st.caption("Scores above 50% flag the patient. Both models are class-balanced, so a score ranks risk rather "
                   "than stating a calibrated probability. Research demonstration only, not for clinical use.")

    st.subheader("What drives the federated score", anchor=False)
    names = [pretty_feature(n) for n in bundle["feature_names"]]
    contrib = x[0] * fed["weights"]
    active = np.nonzero(np.abs(contrib) > 1e-9)[0]
    st.caption("Per-feature contribution (weight × feature value) for this patient, in log-odds. Red pushes risk up, blue pulls it down.")
    st.plotly_chart(charts.coefficients([names[i] for i in active], contrib[active], top=10), key="contrib")
    with st.expander("Global model weights (all patients)"):
        st.plotly_chart(charts.coefficients(names, fed["weights"], top=15), key="global_w")


def page_about() -> None:
    r = get_results()
    m, ds = r["models"], r["dataset"]
    header("About", "About this project", "What SHAFLB is, how it was built, and what it does and does not demonstrate.")
    st.markdown(ABOUT_MD.format(
        n=ds["total_records"], rate=ds["readmission_rate_percent"], feats=r["split"]["features"],
        lr_auc=m["centralized_lr"]["roc_auc"], fed_auc=m["fedavg"]["roc_auc"],
        prox_auc=m["fedprox"]["roc_auc"], dp_auc=m["dp_fedavg"]["roc_auc"], rf_auc=m["centralized_rf"]["roc_auc"],
        hospitals=", ".join(f"{h.replace('_', ' ')} ({v['profile'].lower()})" for h, v in HOSPITALS.items()),
    ))
    st.caption(f"Results generated {r['generated_at']} in {r['runtime_seconds']}s.")


ABOUT_MD = """
### Overview
**SHAFLB (Secure Hospital Analytics using Federated Learning and Blockchain)** shows how several hospitals
can build a shared machine-learning model for **30-day readmission risk in diabetic patients** without
pooling patient records, while keeping a tamper-evident record of how that model was produced.

**Research question.** Can independent hospitals collaboratively train a readmission model that matches a
model trained on pooled data, without exchanging raw records, and can the training process be audited?

### Pipeline
1. **Data.** UCI *Diabetes 130-US Hospitals, 1999–2008* (101,766 encounters). Hospice and expired
   discharges are excluded (readmission is impossible) and only each patient's first encounter is kept,
   giving **{n:,} patients**, of whom **{rate}%** were readmitted within 30 days.
2. **Features.** ICD-9 diagnoses grouped into nine clinical categories; numeric features standardised and
   categoricals one-hot encoded (**{feats} features**). The preprocessor is fit on the training split only.
3. **Hospital network.** The 80% training split is partitioned across five hospitals with different sizes
   and readmission rates (non-IID): {hospitals}. The 20% test set is held by the evaluator.
4. **Centralised baselines.** Class-balanced logistic regression (the upper bound FL is compared with) and a
   random forest, both trained on pooled data.
5. **Federated training.** Each round the server broadcasts the global logistic-regression weights; every
   hospital runs class-weighted mini-batch SGD on its own records; the server aggregates the returned weights
   in proportion to sample counts (**FedAvg**). Variants: **FedProx** (proximal term for non-IID stability)
   and **DP-FedAvg** (update clipping plus Gaussian noise).
6. **Audit ledger.** After every round a block records hospital contributions, test metrics and a SHA-256
   fingerprint of the global model, chained to the previous block's hash. Each hospital keeps a replica;
   a replica that verifies but disagrees with the majority head is rejected.

### Results (held-out test set, ROC-AUC)
| Model | ROC-AUC |
|---|---|
| Centralised logistic regression | {lr_auc:.4f} |
| Centralised random forest | {rf_auc:.4f} |
| FedAvg | {fed_auc:.4f} |
| FedProx | {prox_auc:.4f} |
| DP-FedAvg | {dp_auc:.4f} |

Federated training essentially matches centralised training, and every hospital does better in the
federation than alone. Readmission is hard to predict from administrative data; ROC-AUC around 0.65 is in
line with published results on this dataset.

### Technology
Python 3.11 · NumPy (federated SGD implemented from scratch) · scikit-learn (preprocessing, baselines,
metrics) · pandas · Plotly · Streamlit · SHA-256 via `hashlib`.

### Limitations
- Hospitals are simulated in one process; there is no network transport, authentication or secure aggregation.
- DP noise is illustrative: no privacy accountant is used, so no formal (ε, δ) guarantee is claimed.
- The ledger is replicated with a simple majority vote, not a Byzantine-fault-tolerant consensus protocol.
- The dataset is historical (1999–2008) US data; the model is not validated for clinical use.

### References
- McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.
- Li et al. (2020). *Federated Optimization in Heterogeneous Networks (FedProx).* MLSys.
- McMahan et al. (2018). *Learning Differentially Private Recurrent Language Models.* ICLR.
- Strack et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates.* BioMed Research International.
"""


def page_setup() -> None:
    header("Setup", "Set up the experiment",
           "No results were found in <code>artifacts/</code>. Run the experiment once to generate them.")
    if not RAW_DATA_FILE.exists():
        st.warning(f"The dataset is missing at `{RAW_DATA_FILE.relative_to(RAW_DATA_FILE.parents[2])}`.",
                   icon=":material/dataset:")
        if st.button("Download UCI dataset (~3 MB)", type="primary", icon=":material/download:"):
            with st.spinner("Downloading from archive.ics.uci.edu"):
                download_dataset()
            st.rerun()
        return
    if st.button("Run experiment", type="primary", icon=":material/play_arrow:"):
        run_experiment_with_progress()


# ── navigation ───────────────────────────────────────────────────────────────
PAGES = {
    "overview": st.Page(page_overview, title="Overview", icon=":material/dashboard:", default=True),
    "hospitals": st.Page(page_hospitals, url_path="hospitals", title="Hospital network", icon=":material/local_hospital:"),
    "performance": st.Page(page_performance, url_path="performance", title="Model performance", icon=":material/monitoring:"),
    "federated": st.Page(page_federated, url_path="federated", title="Federated training", icon=":material/hub:"),
    "lab": st.Page(page_lab, url_path="lab", title="FL lab", icon=":material/science:"),
    "ledger": st.Page(page_ledger, url_path="ledger", title="Audit ledger", icon=":material/link:"),
    "simulator": st.Page(page_simulator, url_path="simulator", title="Risk simulator", icon=":material/stethoscope:"),
    "about": st.Page(page_about, url_path="about", title="About", icon=":material/info:"),
}

st.logo(str(LOGO_FILE), size="large", icon_image=str(ICON_FILE))

if artifacts_exist():
    nav = st.navigation({
        "Project": [PAGES["overview"], PAGES["about"]],
        "Analysis": [PAGES["hospitals"], PAGES["performance"], PAGES["federated"]],
        "Interactive": [PAGES["lab"], PAGES["ledger"], PAGES["simulator"]],
    })
    with st.sidebar:
        st.divider()
        res = get_results()
        st.caption(f"Results from {res['generated_at'][:16].replace('T', ' ')} UTC · seed 42")
        if RAW_DATA_FILE.exists() and st.button("Re-run experiment", icon=":material/refresh:", width="stretch"):
            run_experiment_with_progress()
else:
    nav = st.navigation([st.Page(page_setup, title="Setup", icon=":material/build:")])

nav.run()
