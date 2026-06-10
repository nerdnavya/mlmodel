# Normothermic Machine Perfusion — Organ Viability Prediction

> Applying machine learning to real-time organ health monitoring during ex-vivo perfusion.

## Problem
During normothermic machine perfusion, clinicians must manually assess whether an organ is viable for transplant. This is subjective and time-sensitive.

## Approach
- Collected and cleaned multi-parameter sensor data (flow rate, pressure, temperature, metabolic markers)
- Trained classification models (Logistic Regression, Random Forest, XGBoost) to predict viability score
- Evaluated model performance using cross-validation and ROC-AUC

## Tech stack
`Python` `pandas` `scikit-learn` `matplotlib` `Jupyter`

## Results
67% success rate

