import streamlit as st
import numpy as np
import pandas as pd
import polars as pl
import joblib
import traceback
import json
import os

# ---- Page Configuration ----
st.set_page_config(
    page_title="VeriCredit™️ - Advanced Risk Assessment",
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
        /* ... YOUR FULL CSS FROM THE PREVIOUS "BEST INTERFACE" VERSION ... */
        /* Ensure it's the same comprehensive CSS block */
        body, .stApp { color: #EAEAEA; background-color: #0A0F18; font-family: 'Inter', sans-serif; }
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Slab:wght@700&display=swap');
        h1, h2, h3, h4, h5, h6 { color: #FFFFFF; font-weight: 600; }
        h1 { font-family: 'Roboto Slab', serif; letter-spacing: -1px;}
        .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 20px; border-bottom: 2px solid #212834; padding-bottom: 5px; }
        .stTabs [data-baseweb="tab"] { height: 40px; background-color: transparent; padding: 0 12px; border-radius: 6px 6px 0 0; font-weight: 500; color: #7F8C9B; }
        .stTabs [data-baseweb="tab"]:hover { background-color: #1A202C; color: #BDC1C6; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #1A202C; border-bottom: 3px solid #007FFF; color: #E8EAED; }
        .css-1d391kg { background-color: #111620; border-right: 1px solid #212834; }
        .css-1d391kg .stMarkdown p, .css-1d391kg .stMarkdown li, .css-1d391kg .stCaption { color: #9AA0A6; }
        .css-1d391kg .stButton>button { background-color: #212834; color: #BDC1C6; border: 1px solid #313842; width: 100%; font-weight: 500; }
        .css_1d391kg .stButton>button:hover { background-color: #313842; border-color: #4A5568; }
        div[data-testid="stWidgetLabel"] label p { color: #A0AEC0 !important; font-size: 0.875em !important; font-weight: 500; margin-bottom: 0.3rem; }
        .stTextInput > div > div > input, .stNumberInput > div > div > input,
        .stSelectbox > div > div > div, .stMultiSelect > div > div > div > div {
            background-color: #1A202C; color: #E2E8F0; border-radius: 0.375rem;
            border: 1px solid #313842; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            font-size: 0.95em;
        }
        .stSlider > div[data-baseweb="slider"] > div:nth-child(2) > div { background-color: #007FFF; }
        .stSlider > div[data-baseweb="slider"] > div:nth-child(3) { background-color: #4A5568; }
        .stForm .stButton>button {
            background-color: #007FFF; color: white; font-weight: 600; font-size: 1.05em;
            border: none; padding: 0.8em 1.8em; border-radius: 0.375rem;
            transition: background-color 0.2s ease-in-out, transform 0.1s ease, box-shadow 0.2s ease;
            width: 100%;
            box-shadow: 0 2px 4px rgba(0,127,255,0.3);
        }
        .stForm .stButton>button:hover { background-color: #0066CC; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,127,255,0.4); }
        .stForm .stButton>button:active { background-color: #0052A3; transform: translateY(0px); box-shadow: 0 2px 4px rgba(0,127,255,0.3); }
        .streamlit-expanderHeader { font-size: 1.05em; font-weight: 600; color: #BDC1C6; padding: 0.6rem 0.8rem; border-bottom: 1px solid #212834; border-radius: 6px 6px 0 0; background-color: #161A20;}
        .streamlit-expanderContent { background-color: #1A202C; border: 1px solid #212834; border-top: none; border-radius: 0 0 6px 6px; padding: 1.5rem;}
        .prediction-result-card {
            text-align: center; padding: 2.5em; margin: 3em auto; border: 1px solid #2D3748;
            border-radius: 12px; background-color: #161A20;
            box-shadow: 0 12px 24px -4px rgba(0, 0, 0, 0.3), 0 8px 16px -4px rgba(0, 0, 0, 0.2);
            max-width: 700px;
        }
        .prediction-result-title { font-size: 1.2em; color: #718096; margin-bottom: 1.2em; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500; }
        .prediction-result-main-value { font-size: 5em; font-weight: 700; margin-bottom: 0.05em; line-height: 1; text-shadow: 0 0 10px rgba(0,0,0,0.3); }
        .prediction-risk-level { font-size: 1.6em; color: #A0AEC0; margin-top: 0.3em; font-weight: 500; }
        .status-critical-risk { color: #C53030; }
        .status-high-risk { color: #FC8181; }
        .status-moderate-risk { color: #F6AD55; }
        .status-low-risk { color: #48BB78; }
    </style>
""", unsafe_allow_html=True)

# ---- Define Paths & Load Resources ----
MODEL_PATH = 'credit_risk.pkl'
PREPROCESSING_PARAMS_PATH = 'preprocessing_params.json'
FINAL_FEATURES_PATH = 'final_ordered_feature_names.json'
CATEGORICAL_FEATURES_PATH = 'categorical_features_list.json'

@st.cache_resource
def load_all_resources():
    # ... (load_all_resources function same as previous version - no changes needed here for these errors) ...
    resources = {}
    load_success = True
    error_messages = []
    expected_files = {
        "model": MODEL_PATH, "params": PREPROCESSING_PARAMS_PATH,
        "final_cols": FINAL_FEATURES_PATH, "cat_cols": CATEGORICAL_FEATURES_PATH
    }
    for key, path in expected_files.items():
        if not os.path.exists(path):
            error_messages.append(f"CRITICAL 🚨: Required file '{path}' ('{key}') not found. App cannot function.")
            resources[key] = None; load_success = False; continue
        try:
            if path.endswith('.pkl'): resources[key] = joblib.load(path)
            elif path.endswith('.json'):
                with open(path, 'r') as f: resources[key] = json.load(f)
        except Exception as e:
            error_messages.append(f"Error loading '{path}' ('{key}'): {e}")
            resources[key] = None; load_success = False
    if not resources.get("final_cols"):
        error_messages.append("CRITICAL 🚨: `final_feature_names` not loaded or is empty."); load_success = False; resources["final_cols"] = []
    if not resources.get("params"):
        error_messages.append("CRITICAL 🚨: `preprocessing_params` not loaded or is not dictionary."); load_success = False; resources["params"] = {}
    return resources, load_success, error_messages

resources, load_success, load_errors = load_all_resources()

# --- Initialize Session State ---
if 'show_results' not in st.session_state: st.session_state.show_results = False
if 'prediction_proba' not in st.session_state: st.session_state.prediction_proba = None
if 'form_key_counter' not in st.session_state: st.session_state.form_key_counter = 0

# ---- Sidebar ----
# **IMAGE FIX: Assuming you add "logo.png" to your GitHub repo root**
# If you don't have a logo.png, you can comment out or delete the st.sidebar.image line.
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_column_width='auto', caption="VeriCredit™️ Secure Engine")
else:
    st.sidebar.markdown("## 🛡️ VeriCredit™️") # Fallback if logo.png not found

st.sidebar.title("VeriCredit™️ Controls") # Title moved below image/emoji
st.sidebar.info("Leverage advanced AI to assess creditworthiness. Input client data for an instant default risk probability.")
if load_errors:
    for err in load_errors: st.sidebar.error(err)
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Form / New Assessment", key="sidebar_reset_button", use_container_width=True):
    st.session_state.show_results = False
    st.session_state.prediction_proba = None
    st.session_state.form_key_counter += 1
    st.rerun()
st.sidebar.caption("Version: 2.0.1-stable") # Updated version

# ---- Main App Logic ----
st.title("VeriCredit™️ - Advanced Credit Risk Assessment")

if not load_success or resources.get('model') is None or not resources.get('final_cols') or not resources.get('params'):
    st.error("Core application resources failed to load. Please check error messages (sidebar/logs) and ensure all required files (model, preprocessing parameters, feature lists) are present in the GitHub repository and correctly named. The application cannot proceed.")
    st.stop()

model = resources['model']
pp_params = resources['params']
final_feature_names = resources['final_cols']
categorical_features_to_encode = resources.get('cat_cols', [])

# ---- Preprocessing Function ----
def create_feature_vector(ui_inputs, train_params, final_cols_order, cat_cols_to_ohe):
    # ... (Skeleton from before, this is where YOUR DETAILED LOGIC GOES) ...
    polars_data = {}
    polars_data['AMT_INCOME_TOTAL'] = ui_inputs['raw_AMT_INCOME_TOTAL']
    polars_data['DAYS_BIRTH'] = -1 * ui_inputs['raw_AGE_YEARS'] * 365.25
    polars_data['CODE_GENDER'] = ui_inputs['raw_CODE_GENDER']
    polars_data['NAME_EDUCATION_TYPE'] = ui_inputs['raw_NAME_EDUCATION_TYPE']
    polars_data['FLAG_OWN_CAR'] = ui_inputs['raw_FLAG_OWN_CAR']
    polars_data['FLAG_OWN_REALTY'] = ui_inputs['raw_FLAG_OWN_REALTY']
    polars_data['CNT_CHILDREN'] = ui_inputs['raw_CNT_CHILDREN']
    polars_data['NAME_CONTRACT_TYPE'] = ui_inputs['raw_NAME_CONTRACT_TYPE']
    polars_data['AMT_CREDIT'] = ui_inputs['raw_AMT_CREDIT']
    polars_data['AMT_ANNUITY'] = ui_inputs['raw_AMT_ANNUITY']
    polars_data['AMT_GOODS_PRICE'] = ui_inputs.get('raw_AMT_GOODS_PRICE', ui_inputs['raw_AMT_CREDIT'])
    if ui_inputs['raw_YEARS_EMPLOYED'] > 0:
        polars_data['DAYS_EMPLOYED'] = -1 * ui_inputs['raw_YEARS_EMPLOYED'] * 365.25
    else:
        polars_data['DAYS_EMPLOYED'] = 365243
    polars_data['EXT_SOURCE_1'] = ui_inputs.get('raw_EXT_SOURCE_1', np.nan)
    polars_data['EXT_SOURCE_2'] = ui_inputs.get('raw_EXT_SOURCE_2', np.nan)
    polars_data['EXT_SOURCE_3'] = ui_inputs.get('raw_EXT_SOURCE_3', np.nan)
    polars_data['REGION_POPULATION_RELATIVE'] = ui_inputs.get('raw_REGION_POPULATION_RELATIVE', np.nan)
    polars_data['NAME_INCOME_TYPE'] = ui_inputs.get('raw_NAME_INCOME_TYPE')
    polars_data['ORGANIZATION_TYPE'] = ui_inputs.get('raw_ORGANIZATION_TYPE')
    polars_data['NAME_FAMILY_STATUS'] = ui_inputs.get('raw_NAME_FAMILY_STATUS')

    training_stats_defaults = train_params.get("training_column_statistics", {})
    all_original_cols_from_training = set(training_stats_defaults.keys())
    for col in all_original_cols_from_training:
        if col not in polars_data:
             stats_for_col = training_stats_defaults.get(col, {})
             polars_data[col] = stats_for_col.get('median', stats_for_col.get('mode', np.nan))

    initial_schema = {col: pl.Utf8 for col in cat_cols_to_ohe if col in polars_data}
    try:
        df_pl = pl.DataFrame(polars_data, schema_overrides=initial_schema)
    except Exception as e:
        st.error(f"Error creating Polars DataFrame: {e}. Data: {str(polars_data)[:500]}")
        return None

    num_null_cols = train_params.get("numeric_null_cols_from_train", [])
    cat_null_cols = train_params.get("categorical_null_cols_from_train", [])
    impute_num_exprs = [pl.col(c).fill_null(training_stats_defaults.get(c,{}).get('median',0)) for c in num_null_cols if c in df_pl.columns]
    if impute_num_exprs: df_pl = df_pl.with_columns(impute_num_exprs)
    impute_cat_exprs = [pl.col(c).fill_null("Unknown") for c in cat_null_cols if c in df_pl.columns]
    if impute_cat_exprs: df_pl = df_pl.with_columns(impute_cat_exprs)

    cols_to_square = train_params.get("columns_to_square_from_raw", [])
    for col_name in cols_to_square:
        if col_name in df_pl.columns and df_pl[col_name].dtype.is_numeric():
            df_pl = df_pl.with_columns((pl.col(col_name) ** 2).alias(f'{col_name}_squre'))
        else: df_pl = df_pl.with_columns(pl.lit(0.0).alias(f'{col_name}_squre'))

    means = train_params.get("means_for_scaling", {}); stds = train_params.get("stds_for_scaling", {})
    scaling_exprs = []
    for col in df_pl.columns:
        if col in means and col in stds and df_pl[col].dtype.is_numeric():
            scaling_exprs.append(((pl.col(col)-means[col])/(stds[col] if stds[col]!=0 else 1.0)).alias(col))
    if scaling_exprs: df_pl = df_pl.with_columns(scaling_exprs)
            
    df_pandas = df_pl.to_pandas()
    for col in cat_cols_to_ohe:
        if col in df_pandas.columns and not pd.api.types.is_numeric_dtype(df_pandas[col]): # Check if not already numeric
            df_pandas[col] = df_pandas[col].astype(str)
    
    cols_for_ohe_present = [col for col in cat_cols_to_ohe if col in df_pandas.columns]
    if cols_for_ohe_present:
        df_pandas = pd.get_dummies(df_pandas, columns=cols_for_ohe_present, dummy_na=False)
            
    # ** CORRECTED FINAL ASSEMBLY **
    output_df = pd.DataFrame(index=[0], columns=final_cols_order).astype(np.float32) # Create with one row of NaNs and correct dtype
    for col in final_cols_order:
        if col in df_pandas.columns:
            output_df.loc[0, col] = df_pandas[col].astype(np.float32).values[0]
        else:
            output_df.loc[0, col] = 0.0 # If column completely missing after OHE (e.g. a rare category)
    output_df = output_df.fillna(0.0)
    return output_df.values


# ---- Conditional Display: Input Form OR Results Page ----
if not st.session_state.show_results:
    input_config = { # Same input_config as before
        "👤 Applicant Information": {
            "raw_AMT_INCOME_TOTAL": {"label": "Total Annual Income", "type": "number", "min": 20000, "val": 150000, "step": 5000},
            "raw_AGE_YEARS": {"label": "Age (Years)", "type": "number", "min": 18, "max": 75, "val": 35, "step": 1},
            "raw_CODE_GENDER": {"label": "Gender", "type": "select", "options": ["F", "M", "XNA"], "val_idx": 0},
            "raw_NAME_EDUCATION_TYPE": {"label": "Education", "type": "select", "options": ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"], "val_idx": 0},
            "raw_NAME_FAMILY_STATUS": {"label": "Family Status", "type": "select", "options": ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"], "val_idx":0},
            "raw_FLAG_OWN_CAR": {"label": "Owns Car?", "type": "select", "options": ["N", "Y"], "val_idx": 0},
            "raw_FLAG_OWN_REALTY": {"label": "Owns Realty?", "type": "select", "options": ["N", "Y"], "val_idx": 0},
            "raw_CNT_CHILDREN": {"label": "Number of Children", "type": "number", "min":0, "val":0, "step":1},
        },
        "💳 Loan & Employment": {
            "raw_NAME_CONTRACT_TYPE": {"label": "Contract Type", "type": "select", "options": ["Cash loans", "Revolving loans"], "val_idx": 0},
            "raw_AMT_CREDIT": {"label": "Loan Amount Requested", "type": "number", "min": 20000, "val": 250000, "step": 10000},
            "raw_AMT_ANNUITY": {"label": "Loan Annuity", "type": "number", "min": 5000, "val": 20000, "step": 1000},
            "raw_AMT_GOODS_PRICE": {"label": "Goods Price", "type": "number", "min": 0, "val": 225000, "step": 5000, "help": "If goods loan."},
            "raw_NAME_INCOME_TYPE": {"label": "Income Type", "type": "select", "options": ["Working", "Commercial associate", "Pensioner", "State servant"], "val_idx": 0},
            "raw_YEARS_EMPLOYED": {"label": "Years Employed", "type": "number", "val": 5, "min_value": -5, "step": 1, "help": "+ if employed, 0/- for N.A."},
            "raw_ORGANIZATION_TYPE": {"label": "Organization Type (Top 5 + XNA)", "type": "select", "options": ["Business Entity Type 3", "Self-employed", "Other", "School", "Government", "XNA"], "val_idx":0},
        },
        "📊 External Scores & Other": {
            "raw_EXT_SOURCE_1": {"label": "External Source 1", "type": "slider", "min": 0.0, "max": 1.0, "val": 0.5, "step": 0.01},
            "raw_EXT_SOURCE_2": {"label": "External Source 2", "type": "slider", "min": 0.0, "max": 1.0, "val": 0.5, "step": 0.01},
            "raw_EXT_SOURCE_3": {"label": "External Source 3", "type": "slider", "min": 0.0, "max": 1.0, "val": 0.5, "step": 0.01},
            "raw_REGION_POPULATION_RELATIVE": {"label": "Region Pop. Density", "type": "number", "val": 0.0188, "step":0.001, "format":"%.4f"},
        }
    }
    with st.form(key=f"credit_risk_input_form_{st.session_state.form_key_counter}"): # Form key for reset
        st.header("🖋️ Client & Loan Application Details")
        ui_form_inputs = {}
        tabs = st.tabs(list(input_config.keys()))
        for i, (section_name, fields) in enumerate(input_config.items()):
            with tabs[i]:
                for key, config in fields.items():
                    widget_key = f"{key}_{st.session_state.form_key_counter}"
                    if config["type"] == "number": ui_form_inputs[key] = st.number_input(config["label"], min_value=config.get("min"), max_value=config.get("max"), value=config["val"], step=config.get("step"), help=config.get("help"), format=config.get("format"), key=widget_key)
                    elif config["type"] == "select": ui_form_inputs[key] = st.selectbox(config["label"], config["options"], index=config["val_idx"], help=config.get("help"), key=widget_key)
                    elif config["type"] == "slider": ui_form_inputs[key] = st.slider(config["label"], min_value=config["min"], max_value=config["max"], value=config["val"], step=config["step"], help=config.get("help"), key=widget_key)
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🛡️ Assess Credit Risk", use_container_width=True) # CHANGED: use_container_width

        if submitted:
            with st.spinner("⚙️ Deep Analysis In Progress..."):
                try:
                    feature_vector = create_feature_vector(ui_form_inputs, pp_params, final_feature_names, categorical_features_to_encode)
                    if feature_vector is None or not final_feature_names: st.error("Preprocessing error.")
                    elif feature_vector.shape[1] != len(final_feature_names): st.error(f"FATAL: Feature count! Expected {len(final_feature_names)}, Got {feature_vector.shape[1]}.")
                    else:
                        prediction_probabilities = model.predict_proba(feature_vector)
                        st.session_state.prediction_proba = prediction_probabilities[0][1]
                        st.session_state.show_results = True; st.rerun() # CHANGED: st.rerun()
                except Exception as e: st.error(f"Assessment Engine Error: {e}"); st.text(traceback.format_exc())

elif st.session_state.show_results and st.session_state.prediction_proba is not None:
    proba = st.session_state.prediction_proba; proba_percent = proba * 100
    if proba_percent >= 60: risk_level, risk_color_class, risk_icon = "Critical Risk", "status-critical-risk", "🚨"
    elif proba_percent >= 40: risk_level, risk_color_class, risk_icon = "High Risk", "status-high-risk", "⚠️"
    elif proba_percent >= 20: risk_level, risk_color_class, risk_icon = "Moderate Risk", "status-moderate-risk", "🤔"
    else: risk_level, risk_color_class, risk_icon = "Low Risk", "status-low-risk", "✅"
    st.markdown(f"""
        <div class="prediction-result-card">
            <div class="prediction-result-title">Credit Default Risk Assessment</div>
            <div class="prediction-result-main-value {risk_color_class}">{risk_icon} {proba_percent:.1f}%</div>
            <div class="prediction-risk-level">Assessment Outcome: <strong>{risk_level}</strong></div>
        </div>
    """, unsafe_allow_html=True) # Removed inline style, rely on CSS class
    if st.button("‹ Start New Assessment", use_container_width=True, key="new_assessment_button_results"): # CHANGED: use_container_width
        st.session_state.show_results = False; st.session_state.prediction_proba = None;
        st.session_state.form_key_counter += 1 
        st.rerun() # CHANGED: st.rerun()
