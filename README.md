# Generative Model Interpretability and Text Simplification Framework

Master's Thesis Research  
Author: Davinson Poveda  
Core Stack: Python, Transformers, Advanced XAI (SHAP, LIME), SyntaxSHAP, SpaCy, EASSE

---

This framework provides an advanced execution and scientific auditing pipeline to evaluate how generative language models process and simplify complex medical text using the ClaraMeD dataset. 

By blending extrinsic edit-distance metrics with intrinsic Explainable AI (XAI) multi-layer analysis, this toolkit uncovers whether a model perceives complexity based on pure lexical frequency, proximal context, or syntactic burden.

---

## Key Engineering and Research Architectural Modules

The framework is structured across three analytical pillars implemented directly in the core notebook pipelines:

### 1. Dual-Paradigm Inference Engine
* Features an execution pipeline comparing Encoder-Decoder (BART-large-cnn) versus Decoder-Only (LSLlama) architectures.
* Implements memory-efficient 4-bit quantization using bitsandbytes (NF4 configuration with double quantization) for secure GPU batching (Batch Size = 16) with precise reproducibility seeding.

### 2. Multi-Layer XAI Auditor (LIME and SHAP Consolidator)
* Local Sensitivity: Leverages LimeTextExplainer to perform localized word-level perturbations, observing semantic stability against output cosine similarities.
* Cooperative Game Theory: Integrates shap.Explainer to extract deterministic Shapley values, identifying which specific terms act as anchors or blockers during medical text alignments.

### 3. SyntaxSHAP and Structural Complexity (ADD) Engine
* Combines neural attribution values with spaCy dependency parsers to project token weights onto explicit grammatical roles (nsubj, root, amod, appos).
* Calculates the Average Dependency Distance (ADD) for individual tokens to measure the exact correlation between a sentence's structural cognitive load and its model attribution score.

---

## Scientific Evaluation Metrics

Rather than relying on generic observations, the framework implements a custom deterministic validation suite:

* Semantic Fidelity Divergence: Automatically identifies the top-K most complex tokens by highest SHAP weight, isolates/purges them from the original text, and measures the mathematical divergence in cosine similarity before and after the perturbation.
* Physical-to-Semantic Mapping: Unifies physical edit operations (extracted via EASSE word-level operations: REPLACE, DELETE, MOVE, COPY) into a single analytical cross-matrix alongside LIME and SHAP magnitudes.

---

## Getting Started and Replication

### Core Prerequisites
```bash
pip install transformers bitsandbytes accelerate datasets captum shap lime spacy easse sentence-transformers
python -m spacy download es_core_news_sm
