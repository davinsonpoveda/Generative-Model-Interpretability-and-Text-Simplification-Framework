# Generative Model Interpretability and Text Simplification Framework

Master's Thesis Research  
Author: Davinson Poveda  
Core Stack: Python, Transformers, Advanced XAI (SHAP, LIME), SyntaxSHAP, SpaCy, EASSE

[→ Open Interactive Notebook in Google Colab](https://colab.research.google.com/github/davinsonpoveda/Generative-Model-Interpretability-and-Text-Simplification-Framework/blob/main/experimentos_XAI_2026.ipynb)

---
## 🔬 Core Research Question

> **"What does a generative language model actually perceive as 'complex' when processing domain-specific text?"**

This framework provides an advanced execution and scientific auditing pipeline designed to answer this fundamental question. Instead of treating text simplification as a subjective task, this project uses an innovative **Triple XAI Lens** architecture to empirically dissect and measure neural attribution, local sensitivity, and syntactic burden across the ClaraMeD clinical dataset.

---

## 🛠️ The Triple XAI Lens Architecture

The core engineering of this framework monitors model processing behavior through three distinct mathematical angles:

### 1. The Local Sensitivity Lens (LIME)
* Uses `LimeTextExplainer` to perform precise word-level perturbations.
* Measures the local stability of the model by tracking output cosine similarity changes, identifying which contextual shifts trigger variations in prediction.

### 2. The Cooperative Game Theory Lens (SHAP)
* Integrates `shap.Explainer` to compute deterministic Shapley values.
* Isolates the exact mathematical weight of individual clinical terms, classifying whether they act as anchors or processing bottlenecks.

### 3. The Structural & Syntactic Lens (SyntaxSHAP & ADD)
* Projects neural attribution values directly onto explicit grammatical dependencies (e.g., `nsubj`, `root`, `amod`) using `spaCy` tokenization.
* Formulates and tracks the **Average Dependency Distance (ADD)** to mathematically cross-reference the model's cognitive load with the physical distance between related syntactic tokens.

---

## ⚙️ Model & Evaluation Blueprint

* **Dual-Paradigm Inference:** Compares Encoder-Decoder (BART-large-cnn) versus Decoder-Only (LSLlama) setups, utilizing 4-bit quantization (`bitsandbytes` NF4) for secure, memory-efficient GPU execution.
* **Semantic Fidelity Divergence:** Automatically purges the top-K tokens causing the highest SHAP/LIME attribution to physically map and mathematically verify the exact informational divergence before and after structural edits.
* **Physical-to-Semantic Mapping:** Unifies token-level operations (REPLACE, DELETE, COPY, MOVE via `EASSE`) into a structured diagnostic cross-matrix (`df_diagnostico_total`).

---

## 📥 Getting Started & Replication

### Prerequisites
```bash
pip install transformers bitsandbytes accelerate datasets captum shap lime spacy easse sentence-transformers
python -m spacy download es_core_news_sm
