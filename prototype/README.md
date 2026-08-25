# O7 Streamlit Decision-Support Prototype

This folder contains the functional prototype developed for Objective O7 of the MSc thesis **AI-Driven Recommendation System for College Major Selection**.

## Purpose

The application demonstrates how the frozen **Set D + V3 + CatBoost** model can be presented as educational decision support. It does **not** assign a major or replace academic advising. The interface displays the documented held-out performance (**Macro-F1 = 0.2287; accuracy = 0.3137**), returns the top three candidate major categories with model probabilities, and shows a local SHAP explanation for the highest-ranked prediction.

## Run locally

From this folder:

```bash
streamlit run o7_prototype_app.py
```

The repository environment must include Streamlit in addition to the thesis dependencies (CatBoost, SHAP, scikit-learn, imbalanced-learn, pandas, NumPy, Matplotlib, and joblib).

## Included artifacts

- `o7_prototype_app.py` — Streamlit application.
- `Export_Frozen_Model_For_Prototype.ipynb` — reproduces the frozen V3 CatBoost configuration and exports prototype artifacts.
- `catboost_v3_setD.cbm` — exported native CatBoost model.
- `preprocessing_pipeline.joblib` — fitted preprocessing + scaling pipeline; SMOTE is not applied at inference.
- `feature_names.json` — exact 377-predictor input order.
- `feature_defaults.json` — training-set-derived defaults used by manual-entry mode.
- `X4ENTRYMAJ4Y_label_mapping.csv` — encoded target labels.
- `SHAP_top30_dimension_review.csv` — SHAP metadata used for interpretable controls/explanations.
- `HSLS_SetE_feature_audit.csv` — variable descriptions/metadata used for display where available.

## Data-access note

Row-level HSLS:09 data are **not distributed in this public repository**. In particular, `prepared_hsls_setD_4Y.csv` and `demo_students.csv` should remain outside GitHub. The app automatically runs in manual/exploratory mode when `demo_students.csv` is absent. An authorised local copy of `demo_students.csv` may be used for the held-out-student demonstration mode.

## Responsible-use note

The prototype is a research demonstration. The model has modest and uneven fine-grained class performance, its displayed probabilities have not been formally calibrated, and no fairness or external institutional validation has been completed. Outputs should therefore be interpreted as model evidence for discussion, not as a definitive statement of student suitability.
