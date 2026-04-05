# CyberGuard Model Performance & Framework Architecture

This document provides a detailed overview of the core Machine Learning pipelines, performance metrics, framework mappings, and risk-scoring mathematics implemented under `src/model` and `src/detection` in the CyberGuard GRC compliance intelligence system.

## 1. Machine Learning Pipeline Architecture

The primary machine learning engine for checking cyber compliance anomalies avoids heavily-parameterized deep neural networks in favor of a robust, highly stable Scikit-Learn pipeline.

### Natural Language Processing (NLP) Layer
- **Algorithm**: TF-IDF (`TfidfVectorizer`)
- **Capacity**: 3,000 maximum dense mathematical features representation. 
- **Vocabulary Granularity**: Unigram and Bigram combinations (`ngram_range=(1,2)`). Single words (e.g., "authentication") and distinct pairs (e.g., "access control") are encoded.
- **Noise Reduction**: Uses combined Scikit-Learn standard English stop-words coupled with a specialized bespoke suite of GRC stop words (e.g., "shall", "procedure", "ensure", "establish").
- **Term Frequency Dampening**: `sublinear_tf=True` (Scaling applied via logarithmic term frequencies $1 + \log(tf)$ ) to reduce bias on exceedingly lengthy organizational controls.
- **Document Frequency Restraints**: Max document frequency at `85%` (removes boilerplate terms found everywhere) and min frequency at `2` (removes idiosyncratic misspellings).

### Anomaly Detection Engine
- **Algorithm**: Isolation Forest (`IsolationForest` by `scikit-learn`)
- **Structure**: An ensemble model of 200 trees (`n_estimators=200`). This unsupervised mechanism mathematically isolates descriptions that structurally deviate from standard cybersecurity semantics.
- **Contamination Rate**: Hardcoded at `0.1` (10%). The model intrinsically expects 10% of systemic data variation as anomalous boundaries when fitting against the "Gold Standard" baseline.

## 2. Anchored Normalization & Threshold Strategy

A highly specific scaling pipeline has been designed to freeze referential limits and prevent data-drift based on the organization's unique input.
- **NIST-Anchored Scalar**: Normalization relies on `MinMaxScaler(clip=True)` bound exclusively to the minimum and maximum ranges of the `IsolationForest` decision function against the reference framework ONLY. 
- **Drift Prevention**: The `predict.py` logic never refits the normalization curve (`.fit()` explicitly omitted from the pipeline on incoming data, strictly `.transform()`). Normal organizational scores are bounded structurally `[0, 1]`.

## 3. Mathematical Risk Formula Matrix

The model seamlessly transitions the ML probability `(Anomaly Score)` mapping directly into business-logic outputs based on standard GRC Implementation lifecycle gaps.

**Implementation Modifiers (`STATUS_WEIGHT`):**
* `implemented`: `0.1` (Reduces anomaly impact heavily)
* `partial`: `0.5` (Reduces anomaly impact moderately)
* `planned`: `0.7`
* `missing`: `1.0` (Full ML score carried forward)

**Hard Floor Protection (`STATUS_FLOOR`):**
To avoid zeroing risks simply because an organization provided no description, rigorous mathematical floors are guaranteed:
* `implemented`: `0.0`
* `partial`: `2.0`
* `planned`: `4.0`
* `missing`: `7.0` minimum risk value

**Final Computation Functions:**
* **General Status Scoring**: $Risk=max(Floor, min(Anomaly \times Weight \times 10, 10.0))$
* **"Missing" Status Overload Formula**: 
  $Risk_{Missing} = min(7.0 + (Anomaly \times 3.0), 10.0)$
  *This forces 'Missing' controls to map strictly into the [7.0, 10.0] High/Critical tiers while still preserving deterministic variance reflecting how far the gap structurally diverges.*

## 4. Final Risk Classifications

Using the above numerical pipeline computations, deterministic compliance alert thresholds resolve as:

* **CRITICAL RISK:** $[8.0 - 10.0]$
* **HIGH RISK:** $[6.0 - 7.9]$
* **MEDIUM RISK:** $[4.0 - 5.9]$
* **LOW RISK:** $[0.0 - 3.9]$

## 5. Correlated Compliance Frameworks

1. **NIST 800-53 (Baseline Corpus):** 
   * Provides the 984 pure baseline descriptions forming the "Normal" threshold for the IsolationForest's world view.
2. **ISO 27001 (Crosswalk Matrix):** Includes mappings leveraging partial `DIRECT` seeds and inferred matrix mappings representing information security lifecycle phases.
3. **SOC 2 (Crosswalk Matrix):** Maps structural functions to TSP criteria mapped mathematically across the pipeline via matrix analysis tracking cross-walk gaps bridging audit readiness requirements.
