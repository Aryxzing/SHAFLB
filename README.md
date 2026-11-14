# SHAFLB: Secure Hospital Analytics with Federated Learning and Blockchain

[![Tests](https://github.com/Aryxzing/SHAFLB/actions/workflows/tests.yml/badge.svg)](https://github.com/Aryxzing/SHAFLB/actions/workflows/tests.yml)

Five hospitals jointly train a model that predicts **30-day readmission of diabetic patients**
without sharing any patient records. Only model parameters leave a hospital, and every
training round is written to a **tamper-evident, hash-chained audit ledger** that each
hospital keeps a copy of.

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

## Highlights

| | |
|---|---|
| **Federated learning from scratch** | FedAvg, FedProx and DP-FedAvg (clipping + Gaussian noise) implemented in NumPy |
| **Realistic network** | 5 hospitals with non-IID cohort sizes (6k–17k) and readmission rates (4.7%–11.9%) |
| **Honest baselines** | Centralised logistic regression and random forest, plus local-only models per hospital |
| **Audit ledger** | SHA-256 hash chain, replicated per hospital; majority vote rejects rewritten replicas |
| **Interactive dashboard** | FL lab, threshold explorer, tamper lab, explainable risk simulator, light and dark themes |

## Results

Held-out test set of 13,998 patients (8.98% readmitted within 30 days):

| Model | ROC-AUC | PR-AUC | F1 | Recall |
|---|---|---|---|---|
| Centralised logistic regression (pooled data) | 0.6493 | 0.1709 | 0.2209 | 0.5267 |
| Centralised random forest (pooled data) | 0.6484 | 0.1716 | 0.2175 | 0.5449 |
| **FedAvg** (no data shared) | **0.6472** | 0.1676 | 0.2204 | 0.5314 |
| FedProx, μ = 0.05 | 0.6461 | 0.1662 | 0.2212 | 0.5489 |
| DP-FedAvg, clip 1.0, σ = 0.1 | 0.6439 | 0.1660 | 0.2111 | 0.6500 |

- FedAvg retains **99.7%** of the centralised ROC-AUC.
- Every hospital gains **+0.006 to +0.040 ROC-AUC** by joining the federation instead of training alone.
- ROC-AUC around 0.65 is typical for this dataset: readmission is hard to predict from administrative data.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
python run_experiment.py --download
```

```bash
streamlit run app.py
```

The experiment takes about 10 seconds and writes everything the dashboard needs to `artifacts/`.
`--download` fetches the UCI dataset into `data/raw/` if it is missing. The dashboard can also
download the data and run the experiment from its setup screen.

Run the tests with `pytest`. They use synthetic data and never touch `artifacts/`.

## Dashboard

| Page | What it shows |
|---|---|
| Overview | Key numbers, findings and system architecture |
| About | Full project description, methodology, limitations and references |
| Hospital network | Cohort sizes and readmission rates per hospital |
| Model performance | Metrics table, ROC and precision–recall curves, confusion matrices, threshold explorer |
| Federated training | Convergence per round, FedAvg vs FedProx vs DP-FedAvg, value of collaboration, client update norms |
| FL lab | Run a new federated job live: choose algorithm, rounds, hospitals, learning rate and DP noise |
| Audit ledger | Replica status, tamper lab (with and without re-hashing), block explorer, JSON export |
| Risk simulator | Score a patient with both models and see which features drive the federated score |

## How it works

1. **Cohort.** UCI *Diabetes 130-US Hospitals (1999–2008)*, 101,766 encounters. Hospice and expired
   discharges are removed and each patient's first encounter is kept: 69,987 patients.
2. **Features.** ICD-9 codes grouped into nine clinical categories; numeric features standardised,
   categoricals one-hot encoded (154 features). The preprocessor is fit on the training split only.
3. **Partition.** The 80% training split is divided across five hospitals with different sizes and
   class balances. The 20% test set stays with the evaluator.
4. **Federated training.** Each round the server broadcasts the global weights, every hospital runs
   class-weighted mini-batch SGD on its own data, and the server averages the results weighted by
   sample count. FedProx adds a proximal term; DP-FedAvg clips each update and adds Gaussian noise.
5. **Audit.** After each round a block stores contributions, metrics and a SHA-256 fingerprint of the
   model, linked to the previous block's hash. No patient data is ever written to the ledger.

## Project structure

```
app.py                 Streamlit dashboard (8 pages)
run_experiment.py      CLI: download data, run the experiment, print a results table
src/
  config.py            Paths, cohort definition, hospital profiles, FL settings
  data.py              Download, cohort selection, ICD-9 grouping, preprocessing
  models.py            Metrics and centralised baselines
  federated.py         Hospital clients, non-IID partition, FedAvg / FedProx / DP
  blockchain.py        Hash-chained audit ledger and replica consensus
  pipeline.py          End-to-end experiment and artifact persistence
  charts.py            Plotly figures
tests/                 Unit and integration tests (synthetic data)
docs/                  Write-up, presentation guide and screenshots
artifacts/             Generated results, models, ledger (created by run_experiment.py)
```

## Limitations

- Hospitals are simulated in one process: no network transport, authentication or secure aggregation.
- DP-FedAvg is illustrative. There is no privacy accountant, so no formal (ε, δ) guarantee is claimed.
- Replicas are reconciled by simple majority vote, not a Byzantine-fault-tolerant consensus protocol.
- Historical US data; the models are a research demonstration and not validated for clinical use.

## References

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017.
- Li et al., *Federated Optimization in Heterogeneous Networks*, MLSys 2020.
- McMahan et al., *Learning Differentially Private Recurrent Language Models*, ICLR 2018.
- Strack et al., *Impact of HbA1c Measurement on Hospital Readmission Rates*, BioMed Research International, 2014.
