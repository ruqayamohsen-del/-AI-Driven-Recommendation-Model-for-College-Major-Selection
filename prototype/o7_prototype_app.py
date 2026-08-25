"""
O7 Prototype -- University Major Decision-Support Application
================================================================
Generates ranked university-major recommendations from the frozen V3 CatBoost
model trained in this thesis, with a per-student SHAP explanation of WHY those
majors were recommended.

Framed explicitly as decision SUPPORT, not a definitive verdict -- the model's
own documented Macro-F1 (0.2287, held-out) is shown on every page, and outputs
are always a ranked probability list, never a single unqualified answer. This
mirrors the Responsible AI framing already established in the thesis: an
advisory tool for a human advisor to use alongside a student, not a replacement
for one.

Run with:  streamlit run app.py
Expects these files in the same folder as this script (produced by
Export_Frozen_Model_For_Prototype.ipynb):
    catboost_v3_setD.cbm
    preprocessing_pipeline.joblib
    feature_names.json
    demo_students.csv               (optional; local authorised-data demo only)
    X4ENTRYMAJ4Y_label_mapping.csv   (from the SHAP FINAL notebook's output)
    SHAP_top30_dimension_review.csv  (from the SHAP FINAL notebook's output --
                                       used only to pick which ~15 features are
                                       exposed as sliders in manual-entry mode)
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import shap
from catboost import CatBoostClassifier
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Major Recommendation -- Decision Support Prototype", layout="wide")

DOCUMENTED_MACRO_F1 = 0.2287
DOCUMENTED_ACCURACY = 0.3137


# ---------------------------------------------------------------------------
# Load artifacts (cached so the model isn't reloaded on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = CatBoostClassifier()
    model.load_model("catboost_v3_setD.cbm")

    preprocessing = joblib.load("preprocessing_pipeline.joblib")

    with open("feature_names.json") as f:
        feature_names = json.load(f)

    try:
        demo_students = pd.read_csv("demo_students.csv")
    except FileNotFoundError:
        demo_students = None

    try:
        label_map_df = pd.read_csv("X4ENTRYMAJ4Y_label_mapping.csv")
        label_map = dict(zip(label_map_df["Encoded Value"], label_map_df["Label"]))
        # The mapping file was built assuming Encoded Value 0 = "Undeclared/undecided",
        # but some prepared data files keep the raw HSLS code (-1) for that class
        # instead of relabeling it to 0. Cover both so the app works with either.
        if 0 in label_map and -1 not in label_map:
            label_map[-1] = label_map[0]
    except FileNotFoundError:
        label_map = {i: str(i) for i in range(-1, 11)}

    try:
        top30_df = pd.read_csv("SHAP_top30_dimension_review.csv")
        slider_features = top30_df["Variable"].tolist()[:15]
    except FileNotFoundError:
        slider_features = feature_names[:15]

    with open("feature_defaults.json") as f:
        feature_defaults = json.load(f)

    explainer = shap.TreeExplainer(model)

    return model, preprocessing, feature_names, demo_students, label_map, slider_features, feature_defaults, explainer


try:
    model, preprocessing, feature_names, demo_students, label_map, slider_features, feature_defaults, explainer = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Missing artifact: {e}. Run Export_Frozen_Model_For_Prototype.ipynb first and "
        f"place its output files in this folder."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Human-readable variable labels -- optional, degrades gracefully if a file
# is missing. SHAP_top30_dimension_review.csv (already required) covers the
# features shown in the SHAP chart. For full 377-predictor coverage in the
# raw-values table, also copy HSLS_SetE_feature_audit.csv (or the final
# predictor decision table) into this folder.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_variable_labels():
    labels = {}
    for fname in ("HSLS_SetE_feature_audit.csv", "HSLS_final_predictor_decision_table.csv"):
        try:
            df = pd.read_csv(fname)
            if "Variable" in df.columns and "Description" in df.columns:
                labels.update(dict(zip(df["Variable"], df["Description"])))
        except FileNotFoundError:
            pass
    try:
        top30_df = pd.read_csv("SHAP_top30_dimension_review.csv")
        labels.update(dict(zip(top30_df["Variable"], top30_df["HSLS variable description"])))
    except FileNotFoundError:
        pass
    return labels


@st.cache_resource
def load_variable_dimensions():
    """Variable -> plain-language category (Academic / Career interests / Behavioral-psychological /
    Skills-self-efficacy / Demographic-SES), from the thesis's own SHAP dimension analysis.
    Only covers the top-30 file's variables; anything else falls back to 'Other factors'
    at use time -- these are typically the smallest-impact predictors anyway."""
    dims = {}
    try:
        top30_df = pd.read_csv("SHAP_top30_dimension_review.csv")
        dims.update(dict(zip(top30_df["Variable"], top30_df["Feature dimension"])))
    except FileNotFoundError:
        pass
    return dims


variable_labels = load_variable_labels()
variable_dimensions = load_variable_dimensions()


def short_label(var_code: str, max_len: int = 55) -> str:
    """'X2SEX: sample member's sex' if a description exists, else just 'X2SEX'."""
    desc = variable_labels.get(var_code)
    if not desc:
        return var_code
    desc = str(desc).split(".")[0].strip()  # first sentence -- usually the plain-language gist
    if len(desc) > max_len:
        desc = desc[:max_len].rsplit(" ", 1)[0] + "…"
    return f"{var_code}: {desc}"


# ---------------------------------------------------------------------------
# Header -- decision-support framing, shown on every page
# ---------------------------------------------------------------------------
st.title("University Major Recommendation -- Decision Support Prototype")
st.info(
    f"**This is a decision-support tool, not a decision-maker.** The underlying model's "
    f"documented held-out performance is Macro-F1 = {DOCUMENTED_MACRO_F1} "
    f"(Accuracy = {DOCUMENTED_ACCURACY}) on this thesis's 11-category fine-grained major "
    f"task -- a genuinely difficult problem, discussed at length in the thesis. "
    f"Recommendations below are always shown as a ranked, probabilistic list with an "
    f"explanation, never as a single certain answer. This tool is intended to support a "
    f"conversation between a student and an advisor, not to replace one."
)

input_modes = ["Adjust key factors manually (exploratory)"]
if demo_students is not None:
    input_modes.insert(0, "Real held-out student (local authorised data only)")

mode = st.radio(
    "Choose input mode:",
    input_modes,
    horizontal=True,
)

if demo_students is None:
    st.caption(
        "Public-repository mode: no row-level HSLS records are distributed. "
        "The real held-out-student demo becomes available only when an authorised "
        "local demo_students.csv file is present."
    )


# ---------------------------------------------------------------------------
# Build the input row (377 columns, in the exact expected order)
# ---------------------------------------------------------------------------
input_row = None
true_label = None

if mode.startswith("Real held-out"):
    st.subheader("Pick a real student from the held-out test set")
    idx = st.selectbox("Student (anonymous row index -- Set D excludes identifying columns by design)",
                        demo_students.index.tolist())
    row = demo_students.loc[idx]
    true_label = int(row["program_stream"]) if "program_stream" in row else None
    input_row = row[feature_names]

    with st.expander("View this student's raw predictor values"):
        display_df = input_row.to_frame("value")
        display_df.insert(0, "Description", [variable_labels.get(v, "—") for v in display_df.index])
        st.dataframe(display_df)

else:
    st.subheader("Adjust the most influential factors")
    st.caption(
        "Only the ~15 factors with the highest SHAP importance in this thesis's analysis are "
        "exposed here for a manageable demo; every other predictor is fixed at its TRAINING-SET "
        "median (continuous) or mode (discrete) value. This is a simplification for "
        "demonstration purposes only -- it does not change the underlying model or the "
        "thesis's reported results."
    )
    base_row = pd.Series(feature_defaults)

    manual_values = {}
    cols = st.columns(3)
    for i, feat in enumerate(slider_features):
        default = float(base_row.get(feat, 0.0))
        if demo_students is not None and feat in demo_students.columns:
            col_data = demo_students[feat].dropna()
            lo, hi = float(col_data.min()), float(col_data.max())
        else:
            # Public-repository fallback when row-level HSLS demo data are intentionally absent.
            # Use a conservative range around the training-set default solely for interface demonstration.
            span = max(abs(default) * 0.5, 1.0)
            lo, hi = default - span, default + span
        if lo == hi:
            hi = lo + 1.0
        default = min(max(default, lo), hi)
        with cols[i % 3]:
            manual_values[feat] = st.slider(
                short_label(feat, max_len=30), min_value=lo, max_value=hi, value=default,
                help=variable_labels.get(feat),
            )

    input_row = base_row.copy()
    for feat, val in manual_values.items():
        input_row[feat] = val
    input_row = input_row[feature_names]


# ---------------------------------------------------------------------------
# Predict, rank, explain
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Recommendation")

X_input = pd.DataFrame([input_row], columns=feature_names)
X_transformed = preprocessing.transform(X_input)

proba = model.predict_proba(X_transformed)[0]
class_order = model.classes_
ranked = sorted(zip(class_order, proba), key=lambda t: -t[1])

top3 = ranked[:3]
result_df = pd.DataFrame(
    {
        "Rank": [1, 2, 3],
        "Candidate major": [label_map.get(int(c), str(c)) for c, _ in top3],
        "Model confidence": [f"{p:.1%}" for _, p in top3],
    }
)
st.table(result_df.set_index("Rank"))

if true_label is not None:
    st.caption(f"This student's actual reported choice: **{label_map.get(true_label, true_label)}**")

st.subheader("Why this recommendation? (SHAP explanation)")
shap_values = explainer.shap_values(X_transformed)
predicted_class = int(ranked[0][0])
class_idx = list(class_order).index(predicted_class)

if isinstance(shap_values, list):
    sv_for_class = shap_values[class_idx][0]
elif shap_values.ndim == 3:
    sv_for_class = shap_values[0, :, class_idx] if shap_values.shape[2] == len(class_order) else shap_values[0, class_idx, :]
else:
    sv_for_class = shap_values[0]

# --- Category-level breakdown (the plain-language version) ---------------------
# Groups every one of the 377 predictors into the thesis's own dimensions
# (Academic / Career interests / Behavioral-psychological / Skills-self-efficacy /
# Demographic-SES) and shows what share of THIS prediction's total impact each
# category carried -- the "is it mostly grades, interest, skills, or background?"
# answer, before drilling into individual factors below.
dim_contrib = {}
for feat, val in zip(feature_names, sv_for_class):
    dim = variable_dimensions.get(feat, "Other factors")
    dim_contrib[dim] = dim_contrib.get(dim, 0.0) + abs(val)

total_impact = sum(dim_contrib.values())
dim_pct = {d: 100 * v / total_impact for d, v in dim_contrib.items()} if total_impact > 0 else dim_contrib
dim_series = pd.Series(dim_pct).sort_values(ascending=True)

st.caption("**At a glance -- which kind of factor mattered most for this student:**")
fig0, ax0 = plt.subplots(figsize=(9, 3.2))
bar_colors = plt.cm.tab10(np.linspace(0, 1, len(dim_series)))
ax0.barh(dim_series.index, dim_series.values, color=bar_colors)
for i, v in enumerate(dim_series.values):
    ax0.text(v + 0.5, i, f"{v:.0f}%", va="center")
ax0.set_xlabel("Share of this prediction's total SHAP impact (%)")
ax0.set_xlim(0, max(dim_series.values) * 1.2 if len(dim_series) else 1)
plt.tight_layout()
st.pyplot(fig0)
st.caption(
    "This mirrors the thesis's own global dimension breakdown (Academic, Career interests, "
    "Behavioral/psychological, Skills/self-efficacy, Demographic/SES), applied to this one "
    "student's prediction specifically rather than averaged across the whole test set."
)

st.markdown("**Individual factors behind the recommendation:**")

feature_names_transformed = feature_names  # preprocessing here is impute+scale only -- column order preserved
order = np.argsort(-np.abs(sv_for_class))[:10]
top_features = np.array(feature_names_transformed)[order]
top_values = sv_for_class[order]

labels = [short_label(f) for f in top_features]
plot_order = np.argsort(top_values)  # ascending, so the strongest positive bar ends up on top
sorted_labels = np.array(labels)[plot_order]
sorted_values = top_values[plot_order]
colors = ["#1f77b4" if v >= 0 else "#d62728" for v in sorted_values]

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(sorted_labels, sorted_values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("SHAP value (impact on this recommendation)")
plt.tight_layout()
st.pyplot(fig)

with st.expander("View full descriptions for these factors"):
    desc_df = pd.DataFrame({
        "Variable": top_features,
        "Description": [variable_labels.get(f, "—") for f in top_features],
        "SHAP value": top_values,
    }).sort_values("SHAP value", key=abs, ascending=False).set_index("Variable")
    st.dataframe(desc_df)
st.caption(
    "Positive values push toward the top-ranked recommended major; negative values push "
    "away from it. This explains what the MODEL learned from patterns in the data -- it "
    "does not establish that any single factor CAUSES a student to choose a major."
)
