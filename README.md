# Advanced Information Theory Evaluation Framework on MNIST

This repository implements an end-to-end information-theoretic evaluation pipeline to audit deep learning classification profiles on the MNIST handwritten digit dataset. Moving beyond standard accuracy benchmarks, this framework computes exact prediction uncertainty and performance penalties completely from scratch using native mathematical operations.

## 📋 Comprehensive Pipeline Architecture

* **Step 1: Environment Setup & Custom CNN Initialization** 
  Establishes absolute structural reproducibility via seed locking and builds a lightweight Convolutional Neural Network (`SimpleCNN`) optimized to isolate raw evaluation logit streams.
* **Step 2: Manual Shannon Entropy Evaluation** 
  Extracts prediction probabilities via a manual Softmax layer and calculates system uncertainty across all 10,000 test set samples without high-level packages (e.g., SciPy). Includes numeric boundary guards against $\ln(0)$ errors.
* **Step 3: Custom Cross-Entropy Loss Engine** 
  Integrates a custom categorical cross-entropy loss function utilizing the analytical Log-Sum-Exp trick to drive real-time parameter optimization during model training.
* **Step 4: Analytical Visualizations Generation** 
  Exports comparative log-scale histograms, statistical outlier boxplots, and real-time individual classification evaluation traces.
* **Step 5: Confidence vs. Uncertainty Boundary Experiment** 
  Isolates and visualizes the top 5 most confident (minimum entropy) and top 5 most uncertain (maximum entropy) model predictions.

---

## 🔬 Mathematical Formulations & Grounding

### 1. Shannon Entropy $H(p)$
Measures the net operational uncertainty, chaos, or dispersion inherent within the model's final prediction distribution across the 10 target digit classes:
$$H(p) = -\sum_{i=1}^{10} p(x_i) \ln p(x_i)$$
*Measured in **Nats** due to base-$e$ natural log computations.*

### 2. Categorical Cross-Entropy Loss $H(p, q)$
Measures the information distance penalty between the true target distribution $p(x)$ and the model's predicted probability space $q(x)$:
$$H(p, q) = -\sum_{i=1}^{10} p(x_i) \ln q(x_i)$$

### 3. The Analytical Information Link
The theoretical relationship connecting validation performance and prediction uncertainty is governed by the **Kullback-Leibler (KL) Divergence**:
$$H(p, q) = H(p) + D_{KL}(p \parallel q)$$
Because MNIST ground-truth vector profiles are entirely deterministic, one-hot encoded targets, the underlying dataset entropy $H(p)$ is mathematically zero ($H(p) = 0$). Thus, during the optimization phase, cross-entropy reduces cleanly down to the negative log-likelihood of the single correct class index: $H(p, q) = -\ln q_{\text{true\_class}}$.

---

## 📊 Core Experimental Insights

1. **Information Separation:** The generated distribution profiles show that correct predictions group tightly near **0.00 Nats** (high certainty), while misclassifications display a wide spread over higher scales (frequently exceeding **1.50 Nats**).
2. **The Overconfidence Anomaly:** Boxplot analysis isolates specific statistical outliers within the incorrect partition that display near-zero entropy. This proves that due to Softmax logit amplification, a neural network can be confidently incorrect, highlighting why entropy monitoring is critical for safety auditing.
3. **Boundary Tracking:** The extreme entropy boundary extraction successfully filters clean, highly standard handwriting (low entropy) from highly distorted, ambiguous, or noisy strokes (high entropy) without human labels.

---

## 🏃 Execution Instructions

To execute the integrated pipeline and regenerate all performance profiles locally, run the following command:

```bash
python entropy.py
```

### Generated Artifacts
* `step4_distribution_profiles.png`: Log-scale entropy histogram and multi-variable metric variance boxplots.
* `step4_example_traces.png`: Individual validation trace grid mapping confidence, entropy, and loss for baseline inputs.
* `step5_experiment_results.png`: Boundary profile showcasing the most confident vs. most uncertain predictions.
