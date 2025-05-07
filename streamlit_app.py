import streamlit as st
import numpy as np
import pandas as pd
import polars as pl
import joblib
import traceback
import json
import os

# ---- Page Configuration (Same as before) ----
st.set_page_config(
    page_title="IntelliRisk™️ - Creditworthiness Engine 🛡️",
    page_icon="💸", layout="wide", initial_sidebar_state="expanded"
)

# --- Custom CSS (Same as before) ---
st.markdown("""
    <style>
        /* ... YOUR FULL CSS FROM THE PREVIOUS "BEST" VERSION ... */
        body, .stApp { color: #EAEAEA; background-color: #0E1117; font-family: 'Inter', sans-serif; }
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        h1, h2, h3, h4, h5, h6 { color: #FFFFFF; font-weight: 600;}
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] { height: 44px; background-color: transparent; padding: 0 10px;}
        .stTabs [data-baseweb="tab"]:hover { background-color: #222831; border-radius: 4px; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #2A2F3A; border-radius: 4px; }
        .css-1d391kg { background-color: #161A1F; border-right: 1px solid #282C34; }
        .css-1d391kg .stMarkdown p, .css-1d391kg .stMarkdown li { color: #A0AEC0; }
        .css-1d391kg .stButton>button { background-color: #2D3748; color: #E2E8F0; border-color: #4A5568; width: 100%;}
        .css-1d391kg .stButton>button:hover { background-color: #4A5568; border-color: #718096; }
        div[data-testid="stWidgetLabel"] label p { color: #A0AEC0 !important; font-size: 0.9em !important; font-weight: 500; }
        .stTextInput > div > div > input, .stNumberInput > div > div > input,
        .stSelectbox > div > div > div, .stMultiSelect > div > div > div > div {
            background-color: #1A202C; color: #E2E8F0; border-radius: 0.375rem;
            border: 1px solid #4A5568; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        }
        .stSlider > div[data-baseweb="slider"] > div:nth-child(2) > div { background-color: #007ACC; }
        .stButton>button.st-emotion-cache-ódhwd3 { /* Targeting main action button more specifically */
            background-color: #007ACC; color: white; font-weight: 600;
            border: none; padding: 0.75em 1.5em; border-radius: 0.375rem;
            transition: background-color 0.2s ease-in-out, transform 0.1s ease; width: 100%;
        }
        .stButton>button.st-emotion-cache-ódhwd3:hover { background-color: #0062A3; transform: translateY(-1px); }
        .stButton>button.st-emotion-cache-ódhwd3:active { background-color: #005085; transform: translateY(0px); }
        .streamlit-expanderHeader { font-size: 1.1em; font-weight: 500; color: #A0AEC0; }
        .prediction-result-card {
            text-align: center; padding: 2.5em; margin: 2.5em auto; border: 1px solid #2D3748;
            border-radius: 0.75rem; background-color: #1A202C;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            max-width: 650px;
        }
        .prediction-result-title {
            font-size: 1.3em; color: #718096; margin-bottom: 1em;
            text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500;
        }
        .prediction-result-main-value { font-size: 4.5em; font-weight: 700; margin-bottom: 0.1em; line-height: 1; }
        .prediction-risk-level { font-size: 1.5em; color: #A0AEC0; margin-top: 0.2em; font-weight: 500; }
        .status-critical-risk { color: #C53030; }
        .status-high-risk { color: #E53E3E; }
        .status-moderate-risk { color: #DD6B20; }
        .status-low-risk { color: #38A169; }
    </style>
""", unsafe_allow_html=True)


# ---- Define Paths & Load Resources (Same as before) ----
MODEL_PATH = 'credit_risk.pkl'
PREPROCESSING_PARAMS_PATH = 'preprocessing_params.json'
FINAL_FEATURES_PATH = 'final_ordered_feature_names.json'
CATEGORICAL_FEATURES_PATH = 'categorical_features_list.json'

@st.cache_resource
def load_all_resources():
    resources = {}
    load_success = True
    error_messages = []
    expected_files = {
        "model": MODEL_PATH, "params": PREPROCESSING_PARAMS_PATH,
        "final_cols": FINAL_FEATURES_PATH, "cat_cols": CATEGORICAL_FEATURES_PATH
    }
    for key, path in expected_files.items():
        if not os.path.exists(path):
            error_messages.append(f"CRITICAL 🚨: Required file '{path}' for '{key}' not found. App cannot function.")
            resources[key] = None
            load_success = False
            continue
        try:
            if path.endswith('.pkl'): resources[key] = joblib.load(path)
            elif path.endswith('.json'):
                with open(path, 'r') as f: resources[key] = json.load(f)
        except Exception as e:
            error_messages.append(f"Error loading '{path}' for '{key}': {e}")
            resources[key] = None
            load_success = False
    if resources.get("final_cols") is None or not isinstance(resources.get("final_cols"), list) or not resources.get("final_cols"):
        error_messages.append("CRITICAL 🚨: `final_feature_names` not loaded or is empty.")
        load_success = False
        resources["final_cols"] = []
    if resources.get("params") is None or not isinstance(resources.get("params"), dict):
        error_messages.append("CRITICAL 🚨: `preprocessing_params` not loaded or is not a dictionary.")
        load_success = False
        resources["params"] = {}
    return resources, load_success, error_messages

resources, load_success, load_errors = load_all_resources()

# --- Initialize Session State (Same as before) ---
if 'show_results' not in st.session_state: st.session_state.show_results = False
if 'prediction_proba' not in st.session_state: st.session_state.prediction_proba = None

# ---- Sidebar (Same as before) ----
st.sidebar.image("https://i.imgur.com/QpGoN1m.png", use_column_width=True, caption="IntelliRisk™️ Engine")
st.sidebar.title("IntelliRisk™️ Controls")
st.sidebar.info("Assess creditworthiness with our advanced AI. Fill in client data for a default risk probability.")
if load_errors:
    for err in load_errors: st.sidebar.error(err)
st.sidebar.markdown("---")
st.sidebar.caption("Version: 1.0.1-pro")

# ---- Main App Logic ----
st.title("IntelliRisk™️ - Creditworthiness Engine")

if not load_success or resources.get('model') is None or not resources.get('final_cols') or not resources.get('params'):
    st.error("One or more critical application resources failed to load. Please check error messages and application logs. The application cannot proceed.")
    st.stop()

model = resources['model']
pp_params = resources['params']
final_feature_names = resources['final_cols']
categorical_features_to_encode = resources.get('cat_cols', [])

if not st.session_state.show_results:
    st.markdown("🤖 **Engine Ready.** Input client information below to assess credit default risk.", unsafe_allow_html=True)


# ---- Preprocessing Function (Defined globally or at least before the form if it's complex) ----
def create_feature_vector(ui_inputs, train_params, final_cols_order, cat_cols_to_ohe):
    # ... (YOUR FULL, CORRECTED create_feature_vector function from the previous response) ...
    # This function MUST be robust and correctly handle all preprocessing steps
    # based on ui_inputs, train_params, final_cols_order, and cat_cols_to_ohe.
    # It should return a numpy array or None if errors occur.
    # For brevity, I'm not pasting the whole function here again, but use the one we refined.
    # Ensure it correctly uses polars_data, imputation, squaring, scaling, OHE, and final assembly.
    # Example start:
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
    # polars_data['AMT_GOODS_PRICE'] = ui_inputs.get('raw_AMT_GOODS_PRICE', ui_inputs['raw_AMT_CREDIT'])
            
    if ui_inputs['raw_YEARS_EMPLOYED'] > 0:
        polars_data['DAYS_EMPLOYED'] = -1 * ui_inputs['raw_YEARS_EMPLOYED'] * 365.25
    else:
        polars_data['DAYS_EMPLOYED'] = 365243 

    polars_data['EXT_SOURCE_1'] = ui_inputs.get('raw_EXT_SOURCE_1', np.nan) 
    polars_data['EXT_SOURCE_2'] = ui_inputs.get('raw_EXT_SOURCE_2', np.nan)
    polars_data['EXT_SOURCE_3'] = ui_inputs.get('raw_EXT_SOURCE_3', np.nan)
    polars_data['REGION_POPULATION_RELATIVE'] = ui_inputs.get('raw_REGION_POPULATION_RELATIVE', np.nan)
    
    # !!! CRITICAL: Populate polars_data with ALL other base features from training_column_statistics !!!
    # This is where you iterate through features NOT in UI and add them from pp_params
    # Example:
    training_stats_defaults = train_params.get("training_column_statistics", {})
    all_expected_base_cols = set(training_stats_defaults.keys()) # Assuming this covers all original cols
    for col in all_expected_base_cols:
        if col not in polars_data: # If not already set from UI mapping
             stats_for_col = training_stats_defaults.get(col, {})
             polars_data[col] = stats_for_col.get('median', stats_for_col.get('mode', np.nan))


    df_pl = pl.DataFrame([polars_data]) # Now df_pl should have a more complete set of columns

    # 2. Apply `make_nan_free` logic (Imputation)
    num_null_cols = train_params.get("numeric_null_cols_from_train", [])
    cat_null_cols = train_params.get("categorical_null_cols_from_train", [])
    impute_num_exprs = []
    for col in num_null_cols:
        if col in df_pl.columns:
            median_val = train_params.get("training_column_statistics", {}).get(col, {}).get("median", 0)
            impute_num_exprs.append(pl.col(col).fill_null(median_val))
    if impute_num_exprs: df_pl = df_pl.with_columns(impute_num_exprs)
            
    impute_cat_exprs = []
    for col in cat_null_cols:
        if col in df_pl.columns:
            impute_cat_exprs.append(pl.col(col).fill_null("Unknown"))
    if impute_cat_exprs: df_pl = df_pl.with_columns(impute_cat_exprs)

    # 3. Create Squared Features
    cols_to_square = train_params.get("columns_to_square_from_raw", [])
    for col_name in cols_to_square:
        if col_name in df_pl.columns and df_pl[col_name].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            df_pl = df_pl.with_columns((pl.col(col_name) ** 2).alias(f'{col_name}_squre'))
        else:
            df_pl = df_pl.with_columns(pl.lit(0.0).alias(f'{col_name}_squre'))

    # 4. Apply Scaling
    means = train_params.get("means_for_scaling", {})
    stds = train_params.get("stds_for_scaling", {})
    scaling_exprs = []
    for col in df_pl.columns:
        if col in means and col in stds and df_pl[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            mean_val, std_val = means[col], stds[col]
            scaling_exprs.append(((pl.col(col) - mean_val) / (std_val if std_val != 0 else 1.0)).alias(col))
    if scaling_exprs: df_pl = df_pl.with_columns(scaling_exprs)
            
    # 5. Convert to Pandas and One-Hot Encode
    df_pandas = df_pl.to_pandas()
    cols_for_ohe_present = [col for col in cat_cols_to_ohe if col in df_pandas.columns]
    if cols_for_ohe_present:
        df_pandas = pd.get_dummies(df_pandas, columns=cols_for_ohe_present, dummy_na=False)
            
    # 6. Final Assembly
    output_df = pd.DataFrame(columns=final_cols_order)
    for col in final_cols_order:
        if col in df_pandas.columns:
            output_df[col] = df_pandas[col]
        else:
            output_df[col] = 0 # Placeholder - Ideally, this shouldn't happen often if preprocessing is complete
            # print(f"Warning: Final feature '{col}' was expected but not found. Defaulting to 0.")
            
    return output_df.values.astype(np.float32)


# ---- Conditional Display: Input Form OR Results Page ----
if not st.session_state.show_results and model is not None:
    # Define input_config here, right before the form
    input_config = {
        "👤 Applicant Information": {
            "raw_AMT_INCOME_TOTAL": {"label": "Total Annual Income", "type": "number", "min": 20000, "val": 150000, "step": 5000},
            "raw_AGE_YEARS": {"label": "Age (Years)", "type": "number", "min": 18, "max": 75, "val": 35, "step": 1},
            "raw_CODE_GENDER": {"label": "Gender", "type": "select", "options": ["F", "M", "XNA"], "val_idx": 0},
            "raw_NAME_EDUCATION_TYPE": {"label": "Education", "type": "select", "options": ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"], "val_idx": 0},
            "raw_FLAG_OWN_CAR": {"label": "Owns Car?", "type": "select", "options": ["N", "Y"], "val_idx": 0},
            "raw_FLAG_OWN_REALTY": {"label": "Owns Realty?", "type": "select", "options": ["N", "Y"], "val_idx": 0},
            "raw_CNT_CHILDREN": {"label": "Number of Children", "type": "number", "min":0, "val":0, "step":1},
        },
        "💳 Loan & Employment": {
            "raw_NAME_CONTRACT_TYPE": {"label": "Contract Type", "type": "select", "options": ["Cash loans", "Revolving loans"], "val_idx": 0},
            "raw_AMT_CREDIT": {"label": "Loan Amount Requested", "type": "number", "min": 20000, "val": 250000, "step": 10000},
            "raw_AMT_ANNUITY": {"label": "Loan Annuity", "type": "number", "min": 5000, "val": 20000, "step": 1000},
            "raw_AMT_GOODS_PRICE": {"label": "Goods Price", "type": "number", "min": 0, "val": 225000, "step": 5000, "help": "If goods loan."},
            "raw_NAME_INCOME_TYPE": {"label": "Income Type", "type": "select", "options": ["Working", "Commercial associate", "Pensioner", "State servant"], "val_idx": 0}, # Simplified
            "raw_YEARS_EMPLOYED": {"label": "Years Employed", "type": "number", "val": 5, "min_value": -5, "step": 1, "help": "+ if employed, 0/- for N.A."},
            "raw_ORGANIZATION_TYPE": {"label": "Organization Type", "type": "select", "options": ["Business Entity Type 3", "School", "Self-employed", "XNA"], "val_idx":0}, # Simplified
        },
        "📊 External Scores & Other": {
            "raw_EXT_SOURCE_1": {"label": "External Source 1", "type": "slider", "min": 0.0, "max": 1.0, "val": 0.5, "step": 0.01},
            "raw_EXT_SOURCE_2": {"label": "External Source 2", "type": "slider", "min": 0.0, "max": 1.0, "val": 0.5, "step": 0.01},
            "raw_EXT_SOURCE_3": {"label": "External Source 3", "type": "slider", "min": 0.0, "max": 1.0, "val": 0.5, "step": 0.01},
            "raw_REGION_POPULATION_RELATIVE": {"label": "Region Population Density", "type": "number", "val": 0.02, "step":0.001, "format":"%.3f"},
             # Add more inputs for other missing features like FLAG_DOCUMENTs, APARTMENTS_AVG etc. or handle them with defaults in create_feature_vector
        }
    }
    
    with st.form(key="credit_risk_input_form"):
        st.header("🖋️ Client & Loan Application Details")
        ui_inputs_values = {} # To store actual values from widgets
        tabs = st.tabs(list(input_config.keys()))
        for i, (section_name, fields) in enumerate(input_config.items()):
            with tabs[i]:
                for key, config in fields.items():
                    if config["type"] == "number": ui_inputs_values[key] = st.number_input(config["label"], min_value=config.get("min"), max_value=config.get("max"), value=config["val"], step=config.get("step"), help=config.get("help"), format=config.get("format"), key=key) # ADDED KEY
                    elif config["type"] == "select": ui_inputs_values[key] = st.selectbox(config["label"], config["options"], index=config["val_idx"], help=config.get("help"), key=key) # ADDED KEY
                    elif config["type"] == "slider": ui_inputs_values[key] = st.slider(config["label"], min_value=config["min"], max_value=config["max"], value=config["val"], step=config["step"], help=config.get("help"), key=key) # ADDED KEY
        
        st.markdown("---")
        # THE SUBMIT BUTTON IS NOW CORRECTLY INDENTED AND DEFINED
        submitted = st.form_submit_button("🛡️ Assess Credit Risk", use_container_width=True)

        if submitted:
            with st.spinner("🔍 Analyzing application... This may take a moment."):
                try:
                    # ui_inputs_values now holds the actual current values from the form
                    feature_vector = create_feature_vector(ui_inputs_values, pp_params, final_feature_names, categorical_features_to_encode)
                    
                    if feature_vector is None or not final_feature_names:
                        st.error("Critical error in data preprocessing or loading `final_feature_names`.")
                    elif feature_vector.shape[1] != len(final_feature_names):
                        st.error(f"FATAL: Feature count mismatch! Model expects {len(final_feature_names)}, but preprocessor generated {feature_vector.shape[1]}.")
                    else:
                        prediction_probabilities = model.predict_proba(feature_vector)
                        st.session_state.prediction_proba = prediction_probabilities[0][1]
                        st.session_state.show_results = True
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"An error occurred during risk assessment: {e}")
                    st.text(traceback.format_exc())

elif st.session_state.show_results and st.session_state.prediction_proba is not None:
    # ... (Result display code - SAME AS BEFORE) ...
    proba = st.session_state.prediction_proba
    proba_percent = proba * 100
    if proba_percent >= 60: risk_level, risk_color, risk_icon = "Critical Risk", "status-critical-risk", "🚨"
    elif proba_percent >= 40: risk_level, risk_color, risk_icon = "High Risk", "status-high-risk", "⚠️"
    elif proba_percent >= 20: risk_level, risk_color, risk_icon = "Moderate Risk", "status-moderate-risk", "🤔"
    else: risk_level, risk_color, risk_icon = "Low Risk", "status-low-risk", "✅"

    st.markdown(f"""
        <div class="prediction-result-card">
            <div class="prediction-result-title">Credit Default Risk Assessment</div>
            <div class="prediction-result-main-value {risk_color}">
                {risk_icon} {proba_percent:.1f}%
            </div>
            <div class="prediction-risk-level" style="color: var(--{risk_color.replace('status-','')})">
                Assessment: <strong>{risk_level}</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button("‹ Start New Assessment", use_container_width=True, key="new_assessment_button"):
        st.session_state.show_results = False
        st.session_state.prediction_proba = None
        st.experimental_rerun()
