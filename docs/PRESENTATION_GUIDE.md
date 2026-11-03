# SHAFLB Coursework: Presentation & Viva Defense Guide

This document provides a step-by-step **5-Minute Demonstration Script**, **Core Concepts Defense Cheat Sheet**, and **16 Comprehensive Viva Questions & Answers** for presenting the SHAFLB mini-project.

---

## 1. Five-Minute Presentation & Demonstration Script

### Minute 1: Motivation & Architecture (Overview page)
- **Action:** Open Dashboard to **"Overview"**.
- **Script:**
  > *"Good morning. Today I am presenting SHAFLB: Secure Hospital Analytics using Federated Learning and Blockchain. In healthcare, patient records in Electronic Health Records are siloed due to strict privacy regulations like HIPAA and GDPR. Hospitals cannot centralize sensitive records. SHAFLB solves this by training models collaboratively using Federated Learning (FedAvg) while recording all training events on an immutable blockchain audit ledger."*
- **Visual:** Point to the system architecture diagram showing raw patient records remaining private at each hospital node while only model weight updates $\mathbf{w}_k$ are aggregated.

### Minute 2: Dataset & Hospital Heterogeneity (Hospital network page)
- **Action:** Switch to **"Hospital network"**.
- **Script:**
  > *"We evaluated our system on the benchmark UCI Diabetes dataset comprising 69,987 deduplicated patient encounters. The clinical task is predicting 30-day readmission risk (<30 days). We simulated 5 realistic hospital nodes with non-IID data distributions—varying both in patient volume (from 6,000 to 17,000 records) and readmission incidence (from 4.7% to 11.9%) to reflect real-world healthcare heterogeneity."*
- **Visual:** Show the stacked bar chart and readmission rate distribution across Hospital A through Hospital E.

### Minute 3: Centralized Baseline vs. Federated Learning (Model performance & Federated training pages)
- **Action:** Switch to **"Model performance"** then **"Federated training"**.
- **Script:**
  > *"Our centralized Logistic Regression baseline achieves an F1-score of 0.2209 and ROC-AUC of 0.6493 on our held-out test cohort of 13,998 patients. Because the dataset has severe 9:1 class imbalance, we use balanced class weighting to ensure our model achieves ~53% recall of high-risk patients. When we run Federated Learning across 8 communication rounds, the global model converges smoothly, achieving an F1-score of 0.2204 and ROC-AUC of 0.6472."*
- **Visual:** Show the round-by-round convergence plot demonstrating that F1 and ROC-AUC stabilize within 5–8 rounds.

### Minute 4: Direct Model Comparison (Model performance page)
- **Action:** Switch to **"Model performance"** (ROC curve tab).
- **Script:**
  > *"Looking at the direct comparison table, FedAvg achieves 99.77% of centralized F1-score and 99.68% of centralized ROC-AUC. The difference is about 0.3%—demonstrating that hospitals can achieve statistical parity with a centralized model without ever sharing a single raw patient record."*
- **Visual:** Point out the side-by-side metric table and overlayed ROC curves.

### Minute 5: Blockchain Auditing & Live Tamper Test (Audit ledger page)
- **Action:** Switch to **"Audit ledger"**.
- **Script:**
  > *"Finally, we demonstrate our blockchain audit ledger. Each completed FL round creates an immutable block containing participating hospital IDs, sample contributions, metric evaluations, and a SHA-256 hash fingerprint of the global model weights. To demonstrate tamper detection live, let's inject a fake sample count into Block #2..."*
- **Interactive Action:**
  1. In the Tamper lab choose Hospital C, block 2, "Sample count" and click **"Apply attack"**.
  2. Hospital C's status turns **red** (`Broken at #2`) and the block explorer marks block 2 as tampered. Then click **"Restore all replicas"**, switch on **"Cover tracks"** and attack again: the replica now passes its own hash check but is shown as **Outvoted** because its head hash disagrees with the other four hospitals.
  3. Click **"Restore all replicas"** to return every node to consensus (all **green**).
- **Concluding Line:**
  > *"In conclusion, SHAFLB demonstrates that decentralized collaborative learning and cryptographic auditability provide a viable, privacy-preserving paradigm for multi-center clinical AI."*

---

## 2. Core Concepts Defense Cheat Sheet

| Concept | Key Definition & Defense Rationale |
| :--- | :--- |
| **Federated Learning** | A decentralized ML framework where clients train local models on private data and transmit only parameter updates ($\mathbf{w}_k$) to a central aggregator. |
| **FedAvg** | A synchronous federated algorithm where the server averages client model weights proportional to their local dataset sizes: $\mathbf{w}_t = \sum \frac{n_k}{N} \mathbf{w}_k^t$. |
| **Why Centralized ML is Baseline** | Centralized ML assumes all data is pooled into one server. It represents the theoretical upper performance bound with full global gradient information. |
| **Non-IID Data** | Non-Identically and Independently Distributed data, where client datasets differ in feature distributions, class balance, or sample volumes. |
| **Blockchain Role** | Immutable audit trail for provenance, accountability, and tamper detection of training rounds. **Does NOT store patient data and does not provide privacy by itself.** |
| **Class Imbalance** | ~8.98% positive readmission rate. Accuracy alone is misleading because a naive majority classifier achieves ~91% accuracy with 0% clinical utility. |

---

## 3. Comprehensive Viva Questions and Answers

### Q1: What is Federated Learning and why is it needed in healthcare?
**Answer:** Federated Learning (FL) is a distributed machine learning paradigm where multiple clients (e.g., hospitals) collaboratively train a shared global model while keeping their private data decentralized. It is essential in healthcare because patient records are legally protected under regulations like HIPAA and GDPR, preventing hospitals from pooling sensitive medical data into a centralized server.

---

### Q2: Why not simply centralize all hospital data if it yields the best possible model?
**Answer:** Centralizing patient data creates severe regulatory, ethical, and cybersecurity liabilities. It introduces a single point of failure for massive data breaches, violates patient consent frameworks, and faces institutional data-ownership resistance. Federated Learning achieves virtually identical predictive performance while ensuring raw data remains strictly within local institutional firewalls.

---

### Q3: Explain the mathematical formulation of FedAvg.
**Answer:** In communication round $t$, the central server distributes the global weights $\mathbf{w}_{t-1}$. Each hospital $k \in \{1 \dots K\}$ performs local gradient steps on its dataset $\mathcal{D}_k$ of size $n_k$ to produce updated weights $\mathbf{w}_k^t$. The server then computes the weighted average:
$$\mathbf{w}_t = \sum_{k=1}^K \frac{n_k}{N} \mathbf{w}_k^t \quad \text{where } N = \sum_{k=1}^K n_k$$
This ensures that hospitals with larger clinical cohorts contribute proportionally to the global model update.

---

### Q4: Why is weighted averaging preferred over simple unweighted averaging in FedAvg?
**Answer:** In real-world multi-center networks, hospital dataset sizes vary substantially (e.g., 17,000 records vs. 6,000 records). Unweighted averaging would treat a small clinic with high variance equally to a large tertiary medical center with robust sample size, leading to suboptimal convergence and model instability. Sample weighting minimizes the global empirical risk.

---

### Q5: What is non-IID data, and how did you simulate it in this project?
**Answer:** Non-IID (Non-Identically and Independently Distributed) data occurs when clients have different feature or label distributions. We simulated non-IID conditions across 5 hospitals by creating heterogeneous partitions where Hospital A has a higher 30-day readmission incidence (11.1%) and larger volume, whereas Hospital C represents a smaller rural cohort with lower readmission incidence (4.7%).

---

### Q6: Does Federated Learning automatically guarantee complete patient privacy?
**Answer:** No. While FL prevents raw data transmission, model gradients and parameter updates can theoretically be vulnerable to gradient inversion attacks or membership inference in adversarial settings. In production, FL is combined with cryptographic defenses like Differential Privacy (DP-SGD) and Secure Multi-Party Computation (SMPC). In this coursework, we demonstrate architectural data isolation.

---

### Q7: What is the purpose of the Blockchain component in SHAFLB?
**Answer:** The blockchain serves as an **immutable audit ledger**. It records cryptographic fingerprints of global model parameters, participating hospital IDs, contributed sample counts, and evaluation metrics for each round. This establishes verifiable provenance and ensures that training events cannot be retroactively altered, fabricated, or repudiated.

---

### Q8: Does the Blockchain store patient records?
**Answer:** **No, absolutely not.** Storing patient records on a blockchain would violate patient privacy, HIPAA/GDPR compliance, and storage scalability. The blockchain only stores metadata, round summaries, and SHA-256 hashes of model weights.

---

### Q9: Why did you choose Logistic Regression as the primary ML model?
**Answer:** Logistic Regression provides mathematical transparency, clinical interpretability, and direct compatibility with FedAvg parameter averaging without parameter mismatch. Clinicians require interpretable odds ratios for risk factors (e.g., number of inpatient visits, diagnosis categories), making linear models the standard benchmark in clinical risk scoring.

---

### Q10: Why did you evaluate with F1-Score and ROC-AUC instead of Accuracy alone?
**Answer:** The 30-day readmission target is heavily imbalanced (~8.98% positive class). A trivial majority-class model predicting 'No Readmission' for all patients would achieve ~91.02% accuracy but have 0.0 Precision, 0.0 Recall, and 0.50 ROC-AUC (zero clinical utility). F1-Score (harmonic mean of precision and recall) and ROC-AUC measure genuine discriminative capability.

---

### Q11: What do your actual results demonstrate?
**Answer:** Our empirical results show that Centralized Logistic Regression achieves an F1-score of 0.2209 and ROC-AUC of 0.6493, while Federated FedAvg achieves an F1-score of 0.2204 and ROC-AUC of 0.6472. FedAvg retains **99.77% of centralized F1 performance** and **99.68% of ROC-AUC**, proving that decentralized collaborative learning incurs negligible loss in predictive power.

---

### Q12: How does the blockchain detect data tampering?
**Answer:** Each block contains a `previous_hash` pointing to the prior block and its own `block_hash` computed as the SHA-256 hash of its canonical contents. If an adversary alters any value (e.g., sample count or metric score) in a historical block, the recalculated SHA-256 hash no longer matches `block_hash`, and all subsequent `previous_hash` pointers break, immediately alerting the `verify_chain()` function.

---

### Q13: What cohort exclusions did you perform during preprocessing and why?
**Answer:** We applied two key clinical exclusion rules:
1. **Exclusion of hospice/expired discharge dispositions (IDs 11, 13, 14, 19, 20, 21):** Deceased patients cannot be readmitted, so including them introduces target distortion.
2. **First encounter deduplication per patient (`patient_nbr`):** Multiple hospitalizations for the same patient would introduce dependency and potential train/test data leakage.

---

### Q14: How did you group the high-cardinality ICD-9 diagnosis codes?
**Answer:** ICD-9 diagnosis codes (`diag_1`, `diag_2`, `diag_3`) contain hundreds of distinct codes. Following established clinical literature, we mapped them into 9 high-level clinical categories: Diabetes (250.xx), Circulatory (390–459), Respiratory (460–519), Digestive (520–579), Genitourinary (580–629), Neoplasms (140–239), Musculoskeletal (710–739), Injury (800–999), and Other.

---

### Q15: What are the primary limitations of this coursework prototype?
**Answer:**
1. **Simulated Environment:** Hospital clients are simulated as memory-partitioned processes on a single machine rather than physically distributed nodes.
2. **Historical Data:** The UCI Diabetes dataset (1999–2008) is a historical benchmark, not real-time clinical telemetry.
3. **Audit Ledger vs Full Consensus:** Each hospital holds a replica of the SHA-256 chain and replicas are reconciled by simple majority vote, not a Byzantine-fault-tolerant protocol such as PBFT.
4. **Illustrative Differential Privacy:** DP-FedAvg clips and noises updates, but without a privacy accountant there is no formal $(\epsilon, \delta)$ guarantee.

---

### Q16: How could this system be extended for a production hospital deployment?
**Answer:**
1. Implement **Differential Privacy (DP-SGD)** with strict privacy budgeting.
2. Add **Secure Aggregation (SMPC)** to encrypt local updates during transit.
3. Deploy over secure **gRPC/TLS microservices** across institutional Kubernetes clusters.
4. Integrate with **HL7 FHIR** standard EHR connectors for live clinical data ingestion.
5. Deploy a permissioned **Hyperledger Fabric** blockchain consortium for multi-institutional governance.
