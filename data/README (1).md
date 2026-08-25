# Data

This directory documents the data requirements for the MSc thesis:

**AI-Driven Recommendation System for College Major Selection**

The principal dataset used in the thesis is the **High School Longitudinal Study of 2009 (HSLS:09)** from the U.S. National Center for Education Statistics (NCES).

## Data availability

Student-level HSLS:09 source files and prepared analytical datasets are **not redistributed in this public repository**. Users who wish to reproduce the experiments should obtain the appropriate HSLS:09 data directly from NCES and comply with the applicable NCES access, licensing, and redistribution conditions.

The MIT License in this repository applies only to the original project code and does not apply to NCES data or other third-party materials.

## Expected local source file

The final project workflow was developed using the HSLS student-level source file available locally as:

`hsls09_16_student_pets_pear_v1_0.csv`

If your downloaded NCES file has a different name, either rename it locally or update the input path in the data-preparation notebook.

## Data-preparation workflow

Run the notebooks in the following order:

1. `01_HSLS_SetD_Data_Preparation.ipynb`
   - Loads the HSLS source data.
   - Retains valid records for the main entry-major target.
   - Applies the documented predictor audit and selection rules.
   - Produces the final Set D analytical dataset with **12,829 records and 377 predictors**, plus the target.

2. `02_Official_V1_V6_Five_Models.ipynb`
   - Runs the original five-model V1–V6 experimental family.

3. `03_CatBoost_V1_V6_SetD.ipynb`
   - Runs the supplementary CatBoost experiments on Set D.

4. `04_SetE_Sensitivity.ipynb`
   - Evaluates the composite-aware Set E sensitivity analysis using **332 predictors**.

5. `05_HSLS_Binary_Staged_Benchmark.ipynb`
   - Uses the official NCES binary entry-major STEM/non-STEM target.
   - Retains valid `X4ENTMJST` values `{0, 1}`, producing **12,134 records**.
   - Evaluates the information-richness progression across the five feature dimensions.

6. `06_SHAP_Error_Analysis_Final.ipynb`
   - Reproduces the selected Set D + V3 + CatBoost model.
   - Generates the final SHAP and class-level error-analysis outputs.

## Main target variables

### `X4ENTRYMAJ4Y`

The primary thesis target is the official HSLS:09 entry-major variable used for the final **11-category major prediction task**.

The human-readable class mapping used in the final analysis is stored in:

`../documentation/X4ENTRYMAJ4Y_label_mapping.csv`

### `X4ENTMJST`

The staged binary benchmark uses the official NCES STEM/non-STEM entry-major target.

Valid binary values retained for modelling:

- `0` = Non-STEM
- `1` = STEM

The final binary analytical sample contains:

- **8,941 Non-STEM records**
- **3,193 STEM records**
- **12,134 total valid records**

## Predictor documentation

The final Set D predictor audit is stored in:

`../documentation/HSLS_final_predictor_decision_table.csv`

The composite-aware Set E audit is stored in:

`../documentation/HSLS_SetE_feature_audit.csv`

Set D contains **377 retained predictors** organised into five information dimensions:

1. Academic Performance
2. Career Interests and Aspirations
3. Skills and Self-Efficacy
4. Behavioural/Psychological
5. Demographic/SES/Context

## Generated analytical files

The following student-level analytical files may be generated locally during reproduction but should not be committed to the public repository unless redistribution permission is confirmed:

- `prepared_hsls_setD_4Y.csv`
- `prepared_hsls_setE_composite_aware.csv`
- the original HSLS student-level source CSV

These files should remain local and can be excluded through `.gitignore`.

## Reproducibility notes

The final experiments use a fixed random state of **42** and an **80/20 stratified train-test split**. Preprocessing is fitted on training data, and SMOTE is restricted to the training pipeline from V3 onward.

The main multiclass evaluation uses **Macro-F1** as the primary metric. The staged binary benchmark additionally reports accuracy, ROC-AUC, class-level precision/recall/F1, and cross-validation results.

## Important limitation

This repository supports reproduction of the research workflow, but the HSLS analysis is retrospective. The final 377-predictor Set D representation was not designed around a single prospective deployment timestamp. Any future operational advising system should define a fixed prediction time and restrict predictors to information available before that point.
