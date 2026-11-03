# Secure Hospital Analytics using Federated Learning and Blockchain (SHAFLB)
## Undergraduate Machine Learning Coursework Project Report

---

### Abstract
Predicting early hospital readmission risk is critical for improving patient outcomes and reducing unnecessary healthcare expenditures. However, strict data privacy regulations (e.g., HIPAA, GDPR) prevent healthcare institutions from centralizing sensitive Electronic Health Records (EHR) into a shared database. In this project, we implement and evaluate **SHAFLB (Secure Hospital Analytics using Federated Learning and Blockchain)**, a privacy-preserving collaborative framework that enables multiple hospitals to train a global predictive model without exchanging raw patient records. Using the benchmark **UCI Diabetes 130-US Hospitals (1999–2008)** dataset (69,987 deduplicated patient encounters), we simulate a heterogeneous 5-hospital clinical network. We implement the Federated Averaging (FedAvg) algorithm across 8 communication rounds and benchmark it against a centralized Logistic Regression baseline. Furthermore, we develop an immutable SHA-256 blockchain-style audit ledger to provide cryptographic provenance and tamper detection for all federated training events. Our empirical results demonstrate that Federated Learning achieves an F1-score of **0.2204** and ROC-AUC of **0.6472**, retaining **99.77%** and **99.68%** of centralized performance (F1: 0.2209, ROC-AUC: 0.6493) respectively, while ensuring zero raw data centralization.

---

### 1. Introduction
Machine learning in healthcare holds tremendous promise for early risk stratification, preventative interventions, and personalized medicine. Early 30-day readmission is a key clinical quality metric in diabetic inpatient care. Nonetheless, isolated hospital datasets often suffer from limited sample size and demographic skew, while pooling patient records into a centralized cloud repository introduces severe regulatory, legal, and security vulnerabilities.

Federated Learning (FL) offers a transformative paradigm by decentralizing model training: individual hospitals compute model updates on local hardware and transmit only mathematical parameters to a coordinating server. To establish trust, accountability, and reproducibility in decentralized clinical collaborations, blockchain technology provides an immutable, verifiable audit trail of all training rounds.

---

### 2. Problem Statement
1. **The Healthcare Privacy Dilemma:** Medical institutions cannot legally pool identifiable patient health information (PHI) due to stringent data protection mandates.
2. **Clinical Data Heterogeneity:** Hospital patient cohorts are naturally non-identically and independently distributed (non-IID), varying in size, demographic composition, and clinical acuity.
3. **Auditability & Provenance Deficit:** Multi-institutional federated workflows lack verifiable provenance to audit which hospitals participated in specific training rounds, what data volumes were contributed, and whether model updates were tampered with.

---

### 3. Project Objectives
- Formulate a clean clinical cohort from the UCI Diabetes dataset with binary 30-day readmission as the primary target.
- Implement a centralized Machine Learning baseline using Logistic Regression with balanced class weighting.
- Construct a realistic 5-hospital non-IID partition simulating multi-institutional data heterogeneity.
- Implement the Federated Averaging (FedAvg) algorithm with sample-weighted parameter aggregation.
- Build a lightweight SHA-256 blockchain audit ledger with automated integrity verification and tamper detection.
- Provide a clean, interactive Streamlit presentation dashboard and comprehensive test suite for reproducible local execution.

---

### 4. Dataset Description
The project utilizes the **UCI Diabetes 130-US Hospitals (1999–2008)** dataset, comprising 101,766 hospital encounters across 130 US hospitals.

- **Primary Task:** Binary 30-day hospital readmission risk prediction.
- **Target Label Formulation:**
  - Positive Class ($y = 1$): `<30` (readmitted within 30 days post-discharge).
  - Negative Class ($y = 0$): `>30` or `NO` (no early readmission).
- **Cohort Selection Criteria:**
  - **Exclusion of hospice/expired discharge dispositions:** Discharge disposition IDs `11, 13, 14, 19, 20, 21` (representing expired or hospice care) were filtered out because readmission is clinically impossible.
  - **Deduplication:** Retained only the first encounter per unique patient (`patient_nbr`) to ensure independent observations and eliminate inter-encounter leakage across train and test partitions.
- **Cohort Statistics:**
  - **Total Eligible Patients:** 69,987 unique admissions.
  - **Positive Cases (<30 Days):** 6,285 (**8.98%** incidence).
  - **Negative Cases:** 63,702 (**91.02%**).
  - **Train/Test Partition:** Stratified 80/20 split (Train: 55,989 | Test: 13,998).

---

### 5. Data Preprocessing & Feature Engineering
To prevent data leakage, all scalers and encoders were fit strictly on the training set:
1. **ICD-9 Diagnosis Categorization:** High-cardinality primary (`diag_1`), secondary (`diag_2`), and tertiary (`diag_3`) ICD-9 diagnosis codes were grouped into 9 clinical categories:
   - Diabetes (`250.xx`)
   - Circulatory (`390–459`, `785`)
   - Respiratory (`460–519`, `786`)
   - Digestive (`520–579`, `787`)
   - Genitourinary (`580–629`, `788`)
   - Neoplasms (`140–239`)
   - Musculoskeletal (`710–739`)
   - Injury & Poisoning (`800–999`)
   - Other / Missing
2. **Missing Value Imputation:** Dropped unusable administrative fields with extreme missingness (`weight` >95%, `payer_code` >40%, `medical_specialty` >45%). Missing categorical values were encoded as `'Missing'`.
3. **Feature Scaling & Encoding:** Continuous features (`time_in_hospital`, `num_lab_procedures`, `num_medications`, `number_inpatient`, etc.) were standardized with `StandardScaler` (zero mean, unit variance). Categoricals (`race`, `gender`, `age`, medications, diagnoses) were one-hot encoded (`handle_unknown='ignore'`), yielding **154 transformed features**.

---

### 6. System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    SHAFLB FULL SYSTEM ARCHITECTURE                         │
│                                                                            │
│  ┌──────────────┐   Local SGD     Model Parameters (w_k, b_k)              │
│  │  Hospital A  │ ────────────┐                                            │
│  │ (Urban Acad) │             │                                            │
│  └──────────────┘             ▼                                            │
│  ┌──────────────┐       ┌───────────┐       ┌───────────────────────────┐  │
│  │  Hospital B  │ ─────▶│  FedAvg   │──────▶│ Aggregated Global Model   │  │
│  │ (Community)  │       │  Server   │       │ Evaluated on Held-out Set │  │
│  └──────────────┘       └───────────┘       └───────────────────────────┘  │
│  ┌──────────────┐             ▲                           │                │
│  │  Hospital C  │ ────────────┤                           ▼                │
│  │   (Rural)    │             │              ┌──────────────────────────┐  │
│  └──────────────┘             │              │  BLOCKCHAIN AUDIT LEDGER │  │
│  ┌──────────────┐             │              │  Round Block Metadata    │  │
│  │  Hospital D  │ ────────────┤              │  SHA-256 Hash Chaining   │  │
│  │  (Diabetes)  │             │              │  Tamper Detection Check  │  │
│  └──────────────┘             │              └──────────────────────────┘  │
│  ┌──────────────┐             │                                            │
│  │  Hospital E  │ ────────────┘                                            │
│  │  (Veterans)  │                                                          │
│  └──────────────┘                                                          │
│                                                                            │
│  🔒 Privacy Boundary: Raw patient records NEVER leave hospital nodes       │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 7. Centralized Machine Learning Baseline
As the benchmark, a centralized Logistic Regression model was trained on the entire pooled training set ($N = 55,989$). To address the ~9:1 class imbalance, balanced class weighting was utilized:
$$w_c = \frac{N}{2 \cdot N_c}$$
This penalizes false negatives on the minority class ($y=1$), ensuring the model prioritizes identifying high-risk readmissions rather than trivially predicting the majority class.

---

### 8. Simulated Hospital Cohorts (Non-IID Partition)
The training cohort was partitioned across 5 simulated hospital clients with controlled non-IID statistical variation:

| Hospital Node | Institution Profile | Patient Volume ($n_k$) | Readmission Rate (%) | Positive Cases | Negative Cases |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Hospital A** | Urban Academic Medical Center | 17,198 (30.7%) | 11.11% | 1,910 | 15,288 |
| **Hospital B** | Suburban Community Hospital | 14,966 (26.7%) | 8.06% | 1,207 | 13,759 |
| **Hospital C** | Rural Regional Clinic | 8,556 (15.3%) | 4.70% | 402 | 8,154 |
| **Hospital D** | Specialized Diabetes Institute | 9,260 (16.5%) | 11.94% | 1,106 | 8,154 |
| **Hospital E** | Veterans / Military Medical Center | 6,009 (10.7%) | 6.71% | 403 | 5,606 |
| **Total** | Multi-Hospital Consortium | **55,989 (100%)** | **8.98%** | **5,028** | **50,961** |

---

### 9. Federated Learning Algorithm (FedAvg)
We implemented a custom, mathematically transparent Federated Averaging (FedAvg) simulation:

```
Algorithm 1: Federated Averaging (FedAvg) in SHAFLB
-------------------------------------------------------------------------
Input: Communication rounds T=8, Local epochs E=3, Learning rate η=0.05,
       Batch size B=64, Regularization λ=1e-4, K=5 hospital clients.
Output: Final global model parameters (w_T, b_T).

Server initializes global parameters: w_0 = 0, b_0 = 0
For round t = 1, 2, ..., T do:
    For each hospital client k in 1...K in parallel do:
        w_k^t, b_k^t = ClientUpdate(k, w_{t-1}, b_{t-1})
    EndFor
    N = sum_{k=1}^K n_k
    w_t = sum_{k=1}^K (n_k / N) * w_k^t
    b_t = sum_{k=1}^K (n_k / N) * b_k^t
    metrics_t = EvaluateGlobalModel(w_t, b_t, X_test, y_test)
    Blockchain.AppendBlock(round=t, metrics=metrics_t, weights=w_t, bias=b_t)
EndFor
```

---

### 10. Blockchain Audit Ledger Mechanism
The blockchain maintains an immutable record of all federated training events:
1. **Block Structure:**
   - `index`: Monotonically increasing integer (0 = Genesis).
   - `timestamp`: UTC ISO-8601 timestamp.
   - `fl_round`: Communication round number.
   - `participating_hospitals`: List of active hospital node IDs.
   - `hospital_contributions`: Dictionary mapping hospital IDs to contributed sample counts.
   - `total_samples`: Total sample count aggregated in the round.
   - `metrics`: Evaluation metrics dictionary (F1, ROC-AUC, Precision, Recall, Accuracy).
   - `model_weights_hash`: SHA-256 hash of serialized global model weights $\mathbf{w}_t$ and bias $b_t$.
   - `previous_hash`: Cryptographic hash of block $i-1$.
   - `block_hash`: SHA-256 hash of canonical JSON block payload.
2. **Cryptographic Verification (`verify_chain()`):** Sequentially verifies index continuity, verifies that `current.previous_hash == previous.block_hash`, and recalculates SHA-256 hash. If any field in block $j$ is modified, verification fails at block $j$.

---

### 11. Experimental Setup & Hyperparameters
- **Software Stack:** Python 3.11, `scikit-learn` 1.9.1, `pandas` 3.0.5, `numpy` 2.4.6, `matplotlib` 3.11.2, `streamlit` 1.64.0.
- **Reproducibility Seed:** `RANDOM_SEED = 42` across all splits and optimizers.
- **Hardware Execution:** Standard Apple Silicon M-series laptop CPU. Total pipeline runtime: **6.24 seconds**.

---

### 12. Experimental Results & Performance Comparison

#### Primary Performance Metrics (Evaluated on Held-out Test Cohort $N = 13,998$)

| Evaluation Metric | Centralized ML (Pooled Data) | Federated Learning (FedAvg) | Absolute Difference (FL - Central) | Relative Retained Performance |
| :--- | :---: | :---: | :---: | :---: |
| Accuracy | 0.6664 | 0.6625 | -0.0039 | 99.41% |
| Precision (PPV) | 0.1398 | 0.1391 | -0.0007 | 99.50% |
| Recall (Sensitivity) | 0.5267 | 0.5314 | +0.0047 | 100.89% |
| **F1-Score** | **0.2209** | **0.2204** | **-0.0005** | **99.77%** |
| **ROC-AUC (Discrimination)** | **0.6493** | **0.6472** | **-0.0021** | **99.68%** |
| PR-AUC (Average precision) | 0.1709 | 0.1676 | -0.0033 | 98.07% |
| Specificity (TNR) | 0.6802 | 0.6754 | -0.0048 | 99.29% |

#### Federated Learning Convergence History

| FL Round | Contributing Samples | F1-Score | ROC-AUC | Recall | Precision | Accuracy | Block Hash (SHA-256 Prefix) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 55,989 | 0.2106 | 0.6403 | 0.6356 | 0.1262 | 0.5721 | `04f7a84fa57f...` |
| **2** | 55,989 | 0.2190 | 0.6456 | 0.5545 | 0.1364 | 0.6448 | `0708d76b9848...` |
| **3** | 55,989 | 0.2230 | 0.6455 | 0.4726 | 0.1459 | 0.7042 | `5d57dace2619...` |
| **4** | 55,989 | 0.2211 | 0.6469 | 0.5036 | 0.1416 | 0.6813 | `bd36b00a8652...` |
| **5** | 55,989 | 0.2237 | 0.6472 | 0.5465 | 0.1406 | 0.6594 | `c405c3195fff...` |
| **6** | 55,989 | 0.2242 | 0.6481 | 0.4726 | 0.1470 | 0.7064 | `b55ac80eb11a...` |
| **7** | 55,989 | 0.2171 | 0.6477 | 0.6126 | 0.1320 | 0.6034 | `61796f59cbc2...` |
| **8** | 55,989 | 0.2204 | 0.6472 | 0.5314 | 0.1391 | 0.6625 | `fc426295dcc0...` |

Block hashes include a UTC timestamp, so they differ between runs; `artifacts/audit_chain.json` holds the current chain.

#### Extended Experiments

| Model | ROC-AUC | PR-AUC | F1-Score | Recall |
| :--- | :---: | :---: | :---: | :---: |
| Centralized Logistic Regression | 0.6493 | 0.1709 | 0.2209 | 0.5267 |
| Centralized Random Forest | 0.6484 | 0.1716 | 0.2175 | 0.5449 |
| FedAvg | 0.6472 | 0.1676 | 0.2204 | 0.5314 |
| FedProx (μ = 0.05) | 0.6461 | 0.1662 | 0.2212 | 0.5489 |
| DP-FedAvg (clip 1.0, σ = 0.1) | 0.6439 | 0.1660 | 0.2111 | 0.6500 |

**Local-only baselines.** Each hospital trained a class-balanced logistic regression on its own data only (no collaboration), scored on the same test set:

| Hospital | Local-only ROC-AUC | Federated ROC-AUC | Gain from federation |
| :--- | :---: | :---: | :---: |
| Hospital A | 0.6382 | 0.6472 | +0.0090 |
| Hospital B | 0.6415 | 0.6472 | +0.0057 |
| Hospital C | 0.6114 | 0.6472 | +0.0358 |
| Hospital D | 0.6265 | 0.6472 | +0.0207 |
| Hospital E | 0.6071 | 0.6472 | +0.0401 |

Every hospital obtains a better model by joining the federation than by training alone, with the smallest hospitals gaining most.

---

### 13. Discussion & Clinical Implications
1. **Statistical Parity:** FedAvg achieves **99.77% of centralized F1-score** and **99.68% of centralized ROC-AUC**. The tiny difference (about 0.3%) confirms that decentralizing clinical data across institutional silos does not compromise predictive quality.
2. **Handling Imbalance:** Under a severe 9:1 class imbalance, naive accuracy is uninformative. Balanced weighting allows both centralized and federated models to achieve ~53% recall, successfully detecting the majority of high-risk early readmission cases.
3. **Auditability Without Privacy Compromise:** The blockchain records all round parameters and hashes without storing raw patient records. The live tampering test demonstrated 100% detection of corrupted metadata.

---

### 14. Limitations
- **Simulated Federation:** The 5 hospital nodes run as local memory partitions on a single workstation rather than physically distributed network servers.
- **Historical EHR Cohort:** The UCI Diabetes dataset represents historical clinical practices (1999–2008) and is utilized for academic validation rather than active clinical deployment.
- **Prototype Ledger:** Each hospital holds a replica of the SHA-256 hash chain and replicas are reconciled by majority vote; this is not a Byzantine-fault-tolerant consensus protocol.
- **Cryptographic Defenses:** DP-FedAvg (update clipping plus Gaussian noise) is implemented for illustration without a privacy accountant, so no formal $(\epsilon, \delta)$ guarantee is claimed. Secure aggregation (SMPC) remains future work.

---

### 15. Future Work
- Implementation of **$(\epsilon, \delta)$-Differential Privacy (DP-SGD)** with formal privacy budgeting.
- Integration of **Secure Multi-Party Computation (SMPC)** for encrypted gradient aggregation.
- Real-time **HL7 FHIR API** connectors for live EHR data streams.
- Deployment of a permissioned **Hyperledger Fabric** consortium across institutional healthcare clusters.

---

### 16. Conclusion
The SHAFLB coursework project successfully demonstrates that multiple healthcare institutions can collaboratively train an accurate diabetic readmission prediction model using Federated Learning (FedAvg) without centralizing sensitive patient records. By integrating a lightweight blockchain audit ledger, SHAFLB ensures end-to-end cryptographic provenance and tamper evidence. The experimental results prove that privacy-preserving federated machine learning achieves near-exact parity with centralized learning, providing a practical blueprint for secure multi-hospital AI collaboration.
