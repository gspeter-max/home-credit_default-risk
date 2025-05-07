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
    page_title="IntelliRisk™️ - Creditworthiness Engine 🛡️",
    page_icon="💸", layout="wide", initial_sidebar_state="expanded"
)

# --- Custom CSS (Same as your preferred version) ---
st.markdown("""
    <style>
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
        .streamlit-expanderHeader { font-size: 1.1em; font-weight: 500; color: #A0AEC0; } /* Lighter for better visibility */
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
        .status-critical-risk { color: #C53030; } /* Even darker red */
        .status-high-risk { color: #E53E3E; }
        .status-moderate-risk { color: #DD6B20; }
        .status-low-risk { color: #38A169; }
    </style>
""", unsafe_allow_html=True)

# ---- Define Paths for ALL Artifacts ----
MODEL_PATH = 'credit_risk.pkl'
PREPROCESSING_PARAMS_PATH = 'preprocessing_params.json'
FINAL_FEATURES_PATH = 'final_ordered_feature_names.json'
CATEGORICAL_FEATURES_PATH = 'categorical_features_list.json'

@st.cache_resource # Cache all resources together
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
            print(f"Successfully loaded: {path} for {key}")
        except Exception as e:
            error_messages.append(f"Error loading '{path}' for '{key}': {e}")
            resources[key] = None
            load_success = False
    return resources, load_success, error_messages

resources, load_success, load_errors = load_all_resources()

# --- Initialize Session State ---
if 'show_results' not in st.session_state: st.session_state.show_results = False
if 'prediction_proba' not in st.session_state: st.session_state.prediction_proba = None

# ---- Sidebar ----
st.sidebar.image("https://i.imgur.com/QpGoN1m.png", use_column_width=True, caption="IntelliRisk™️ Engine") # Example custom image URL
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
    st.stop() # Halt execution if critical resources are missing

model = resources['model']
pp_params = resources['params']
final_feature_names = resources['final_cols']
categorical_features_to_encode = resources.get('cat_cols', [])


if not st.session_state.show_results:
    st.markdown("🤖 **Engine Ready.** Input client information below to assess credit default risk.", unsafe_allow_html=True)

if not st.session_state.show_results:
    with st.form(key="credit_risk_input_form"):
        st.header("🖋️ Client & Loan Application Details")
        # --- Define Input Fields using a dictionary for easier management ---
        # YOU NEED TO EXPAND THIS with ALL relevant UI inputs for your model
        # Default values should be sensible. Help texts are useful.
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
                "raw_YEARS_EMPLOYED": {"label": "Years Employed", "type": "number", "val": 5, "min_value": -5, "step": 1, "help": "+ for employed, 0 or - for unemployed/N.A."},
            },
            "📊 External Scores & Other": {
                "raw_EXT_SOURCE_1": {"label": "External Source 1", "type": "slider", "min": 0.0, "max": 1.0, "val": 0.5, "step": 0.01, "help": "Normalized score from external data source 1 (0-1)"},
                "raw_EXT_SOURCE_2": {"label": "External Source 2", "type": "slider", "min": 0.0, "max": 1.0, "val": 0.5, "step": 0.01},
                "raw_EXT_SOURCE_3": {"label": "External Source 3", "type": "slider", "min": 0.0, "max": 1.0, "val": 0.5, "step": 0.01},
                # Add more relevant inputs from your list of warnings
                "raw_REGION_POPULATION_RELATIVE": {"label": "Region Population Density", "type": "number", "val": 0.02, "step":0.001, "format":"%.3f", "help":"Normalized population of region where client lives"},
            }
        }
        raw_inputs_from_ui = {}
        tabs = st.tabs(list(input_config.keys()))
        for i, (section_name, fields) in enumerate(input_config.items()):
            with tabs[i]:
                for key, config in fields.items():
                    if config["type"] == "number": raw_inputs_from_ui[key] = st.number_input(config["label"], min_value=config.get("min"), max_value=config.get("max"), value=config["val"], step=config.get("step"), help=config.get("help"), format=config.get("format"))
                    elif config["type"] == "select": raw_inputs_from_ui[key] = st.selectbox(config["label"], config["options"], index=config["val_idx"], help=config.get("help"))
                    elif config["type"] == "slider": raw_inputs_from_ui[key] = st.slider(config["label"], min_value=config["min"], max_value=config["max"], value=config["val"], step=config["step"], help=config.get("help"))
        
        st.markdown("---")

        def create_feature_vector(ui_inputs, train_params, final_cols_order, cat_cols_to_ohe):
            # 1. Prepare initial Polars DataFrame from UI and defaults
            polars_data = {}
            # Map UI inputs (keys like 'raw_AMT_INCOME_TOTAL') to actual feature names
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
            # AMT_GOODS_PRICE might come from UI or be derived. For now, assume it's a base feature.
            # polars_data['AMT_GOODS_PRICE'] = ui_inputs.get('raw_AMT_GOODS_PRICE', ui_inputs['raw_AMT_CREDIT']) # Example if it can default to AMT_CREDIT
            
            if ui_inputs['raw_YEARS_EMPLOYED'] > 0:
                polars_data['DAYS_EMPLOYED'] = -1 * ui_inputs['raw_YEARS_EMPLOYED'] * 365.25
            else:
                polars_data['DAYS_EMPLOYED'] = 365243 # Common placeholder

            polars_data['EXT_SOURCE_1'] = ui_inputs.get('raw_EXT_SOURCE_1', np.nan) # Handle if not in UI
            polars_data['EXT_SOURCE_2'] = ui_inputs.get('raw_EXT_SOURCE_2', np.nan)
            polars_data['EXT_SOURCE_3'] = ui_inputs.get('raw_EXT_SOURCE_3', np.nan)
            polars_data['REGION_POPULATION_RELATIVE'] = ui_inputs.get('raw_REGION_POPULATION_RELATIVE', np.nan)
            
            # !!! YOU MUST ADD ALL OTHER BASE FEATURES HERE FROM `application_train.csv` !!!
            # For features not in UI, impute using medians/modes from `train_params['training_column_statistics']`
            # Example:
            # base_cols_not_in_ui = set(train_params['training_column_statistics'].keys()) - set(polars_data.keys())
            # for col in base_cols_not_in_ui:
            #    stats = train_params['training_column_statistics'].get(col, {})
            #    polars_data[col] = stats.get('median', stats.get('mode', np.nan)) # Prioritize median, then mode, then NaN

            df_pl = pl.DataFrame([polars_data])

            # 2. Apply `make_nan_free` logic (Imputation)
            num_null_cols = train_params.get("numeric_null_cols_from_train", [])
            cat_null_cols = train_params.get("categorical_null_cols_from_train", [])
            impute_num_exprs = []
            for col in num_null_cols:
                if col in df_pl.columns: # Impute only if column exists in current df_pl
                    # Use median from training data (MUST BE SAVED in train_params['training_column_statistics'][col]['median'])
                    median_val = train_params.get("training_column_statistics", {}).get(col, {}).get("median", 0) # Default to 0 if not found
                    impute_num_exprs.append(pl.col(col).fill_null(median_val))
            if impute_num_exprs: df_pl = df_pl.with_columns(impute_num_exprs)
            
            impute_cat_exprs = []
            for col in cat_null_cols:
                if col in df_pl.columns:
                    impute_cat_exprs.append(pl.col(col).fill_null("Unknown"))
            if impute_cat_exprs: df_pl = df_pl.with_columns(impute_cat_exprs)


            # 3. Create Squared Features (from raw/imputed values, before scaling)
            cols_to_square = train_params.get("columns_to_square_from_raw", [])
            for col_name in cols_to_square:
                if col_name in df_pl.columns and df_pl[col_name].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]: # Ensure numeric
                    df_pl = df_pl.with_columns((pl.col(col_name) ** 2).alias(f'{col_name}_squre'))
                else: # If col not present or not numeric, add placeholder for squared version
                    df_pl = df_pl.with_columns(pl.lit(0.0).alias(f'{col_name}_squre'))


            # 4. Apply `make_efficient` logic (Scaling)
            means = train_params.get("means_for_scaling", {})
            stds = train_params.get("stds_for_scaling", {})
            scaling_exprs = []
            for col in df_pl.columns: # Scale all numeric columns that have stats
                if col in means and col in stds and df_pl[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    mean_val, std_val = means[col], stds[col]
                    scaling_exprs.append(((pl.col(col) - mean_val) / (std_val if std_val != 0 else 1.0)).alias(col))
            if scaling_exprs: df_pl = df_pl.with_columns(scaling_exprs)
            
            # 5. Convert to Pandas and One-Hot Encode specified categorical features
            df_pandas = df_pl.to_pandas()
            cols_for_ohe_present = [col for col in cat_cols_to_ohe if col in df_pandas.columns]
            if cols_for_ohe_present:
                df_pandas = pd.get_dummies(df_pandas, columns=cols_for_ohe_present, dummy_na=False) # Ensure dummy_na strategy matches training
            
            # 6. Final Assembly: Ensure all features from final_cols_order are present
            output_df = pd.DataFrame(columns=final_cols_order) # Create empty df with correct columns and order
            for col in final_cols_order:
                if col in df_pandas.columns:
                    output_df[col] = df_pandas[col]
                else:
                    # This is where you handle features that were in training but not generated now
                    # (e.g. an OHE category not present in this specific input, or a feature entirely missed)
                    # Defaulting to 0 is a simple strategy but might be suboptimal.
                    output_df[col] = 0
                    print(f"Warning: Final feature '{col}' was expected but not found in processed data. Defaulting to 0.")
            
            return output_df.values.astype(np.float32)


        submitted = st.form_submit_button("🛡️ Assess Credit Risk", key="submit_assessment")

        if submitted:
            with st.spinner("🔍 Analyzing application... This may take a moment."):
                try:
                    feature_vector = create_feature_vector(raw_inputs_from_ui, pp_params, final_feature_names, categorical_features_to_encode)
                    if feature_vector is None or not final_feature_names:
                        st.error("Critical error in data preprocessing or loading `final_feature_names`.")
                    elif feature_vector.shape[1] != len(final_feature_names):
                        st.error(f"FATAL: Feature count mismatch! Model expects {len(final_feature_names)}, but preprocessor generated {feature_vector.shape[1]}. "
                                 "This means `final_ordered_feature_names.json` or the `create_feature_vector` logic is not aligned with the training.")
                    else:
                        prediction_probabilities = model.predict_proba(feature_vector)
                        st.session_state.prediction_proba = prediction_probabilities[0][1]
                        st.session_state.show_results = True
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"An error occurred during risk assessment: {e}")
                    st.text(traceback.format_exc())

elif st.session_state.show_results and st.session_state.prediction_proba is not None:
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
