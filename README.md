# Fairness and Bias Auditing in Financial ML Models

## Overview

This repository contains a Python script, `fairness_and_bias_auditing.py`, designed to illustrate and audit algorithmic bias in machine learning models for financial applications, such as credit scoring and loan approvals.

The script demonstrates how even simple models can propagate biases against protected groups (e.g., race, gender), and provides a minimal example of fairness auditing using commonly accepted metrics.

---

## Key Features

* 📊 **Synthetic data generation**: Creates a dataset with an explicit protected attribute and embedded bias to simulate real-world unfairness.
* 🤖 **Logistic regression model**: Simple, interpretable baseline classifier for demonstration.
* 🧮 **Group-wise metrics**: Computes accuracy, true positive rate (TPR), and positive prediction rates separately for each group.
* ⚖️ **Disparate impact analysis**: Measures fairness in terms of positive outcome ratios between groups.
* 🚨 **Bias flagging**: Detects and warns if disparate impact falls outside an acceptable legal/ethical range (commonly 0.8 to 1.25).

---

## How it works

1. **Generate biased synthetic data**
   The dataset simulates a scenario where the probability of receiving a positive label (e.g., loan approval) is lower for a certain protected group.

2. **Train/Test split**
   Uses a random 70/30 split to evaluate generalization.

3. **Train classifier**
   A logistic regression model is fit on the training data.

4. **Evaluate fairness**

   * Calculate overall accuracy.
   * Compute group-specific accuracy, TPR, and positive rates.
   * Compute disparate impact ratio.
   * Print warnings if bias is detected.

---

## Example Output

```
Overall Accuracy: 0.75

Group-wise metrics:
   group  accuracy   TPR  positive_rate
0      0      0.73  0.65          0.46
1      1      0.77  0.71          0.41

Disparate Impact (group 1 / group 0): 0.89
✅ No major disparate impact detected.
```

---

## Usage

```bash
python fairness_and_bias_auditing.py
```

Ensure that you have the required dependencies installed:

```bash
pip install numpy pandas scikit-learn
```

---

## Extensions

This example can be extended to include:

* Bootstrap confidence intervals for group metrics.
* Advanced metrics (e.g., equal opportunity difference, average odds difference).
* Distributional comparisons (e.g., Wasserstein distance).
* Mitigation strategies (e.g., reweighting, adversarial debiasing).

---

## Disclaimer

This script is for educational and research demonstration purposes only. The simplified data generation and metrics are designed to illustrate fairness principles but do not represent full production-level compliance checks.

---

## License

MIT License

---

## Author

\[Prefrontal Corporate]

---

## References

* Feldman et al., "Certifying and removing disparate impact", KDD, 2015.
* Hardt et al., "Equality of opportunity in supervised learning", NeurIPS, 2016.
* Barocas, Hardt, and Narayanan, "Fairness and Machine Learning", fairmlbook.org.
