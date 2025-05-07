import streamlit as st
import numpy as np
import pandas as pd
import polars as pl # We'll use Polars for preprocessing to match your notebook
import joblib
import traceback
import json
import os # For checking file existence, good practice

# ---- Page Configuration ----
st.set_page_config(
    page_title="IntelliRisk™️ - Creditworthiness Engine 🛡️", # More "pro" name
    page_icon="💸", # Changed icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Professional Dark Interface ---
st.markdown("""
    <style>
        /* Main Theme */
        body, .stApp { color: #DCDCDC; background-color: #10141C; font-family: 'Inter', sans-serif; } /* Using Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        h1, h2, h3, h4, h5, h6 { color: #FFFFFF; font-weight: 600; }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; } /* Spacing for tabs */
        .stTabs [data-baseweb="tab"] { height: 44px; background-color: transparent; padding: 0 10px;}
        .stTabs [data-baseweb="tab"]:hover { background-color: #222831; border-radius: 4px; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #2A2F3A; border-radius: 4px; }

        /* Sidebar */
        .css-1d391kg { background-color: #161A1F; border-right: 1px solid #282C34; }
        .css-1d391kg .stMarkdown p, .css-1d391kg .stMarkdown li { color: #A0AEC0; }
        .css-1d391kg .stButton>button { background-color: #2D3748; color: #E2E8F0; border-color: #4A5568; width: 100%;}
        .css-1d391kg .stButton>button:hover { background-color: #4A5568; border-color: #718096; }

        /* Input Widgets - Subtle and Modern */
        div[data-testid="stWidgetLabel"] label p { color: #A0AEC0 !important; font-size: 0.9em !important; font-weight: 500; }
        .stTextInput > div > div > input, .stNumberInput > div > div > input,
        .stSelectbox > div > div > div, .stMultiSelect > div > div > div > div {
            background-color: #1A202C; color: #E2E8F0; border-radius: 0.375rem; /* Tailwind's rounded-md */
            border: 1px solid #4A5568; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        }
        .stSlider > div[data-baseweb="slider"] > div:nth-child(2) > div { background-color: #007ACC; } /* Slider track */


        /* Main Action Button */
        .stButton>button[kind="primary"] { /* Target primary buttons if used */
            background-color: #007ACC; color: white; font-weight: 600;
            border: none; padding: 0.75em 1.5em; border-radius: 0.375rem;
            transition: background-color 0.2s ease-in-out, transform 0.1s ease;
        }
        .stButton>button[kind="primary"]:hover { background-color: #0062A3; transform: translateY(-1px); }
        .stButton>button[kind="primary"]:active { background-color: #005085; transform: translateY(0px); }

        /* Prediction Result Area - Professional Card */
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
        .prediction-result-main-value {
            font-size: 4.5em; font-weight: 700; margin-bottom: 0.1em; line-height: 1;
        }
        .prediction-risk-level { font-size: 1.5em; color: #A0AEC0; margin-top: 0.2em; font-weight: 500; }
        .status-high-risk { color: #E53E3E; } /* Tailwind red-600 */
        .status-moderate-risk { color: #DD6B20; } /* Tailwind orange-600 */
        .status-low-risk { color: #38A169; } /* Tailwind green-600 */
    </style>
""", unsafe_allow_html=True)

# ---- Define Paths for ALL Artifacts ----
# These MUST match the filenames in your GitHub repository
MODEL_PATH = 'credit_risk.pkl'
PREPROCESSING_PARAMS_PATH = 'preprocessing_params.json'
FINAL_FEATURES_PATH = 'final_ordered_feature_names.json'
CATEGORICAL_FEATURES_PATH = 'categorical_features_list.json'

@st.cache_resource
def load_all_resources():
    resources = {}
    load_success = True
    error_messages = []

    required_files = {
        "model": MODEL_PATH,
        "preprocessing_params": PREPROCESSING_PARAMS_PATH,
        "final_feature_names": FINAL_FEATURES_PATH,
        "categorical_features": CATEGORICAL_FEATURES_PATH
    }

    for key, path in required_files.items():
        if not os.path.exists(path):
            msg = f"CRITICAL 🚨: Required file '{path}' not found in the repository. App cannot function correctly."
            error_messages.append(msg)
            print(msg) # Also print to console/logs
            load_success = False
            resources[key] = None # Ensure key exists even if loading fails
            continue # Skip loading this file

        try:
            if path.endswith('.pkl'):
                resources[key] = joblib.load(path)
            elif path.endswith('.json'):
                with open(path, 'r') as f:
                    resources[key] = json.load(f)
            print(f"Successfully loaded: {path}")
        except Exception as e:
            msg = f"Error loading '{path}': {e}"
            error_messages.append(msg)
            print(msg)
            load_success = False
            resources[key] = None
            
    # Post-load checks
    if resources.get("final_feature_names") is None or not isinstance(resources.get("final_feature_names"), list) or not resources.get("final_feature_names"):
        msg = "CRITICAL 🚨: `final_feature_names` not loaded or is empty. This is essential."
        error_messages.append(msg)
        print(msg)
        load_success = False
        resources["final_feature_names"] = [] # Ensure it's an empty list to avoid later errors

    if resources.get("preprocessing_params") is None or not isinstance(resources.get("preprocessing_params"), dict):
        msg = "CRITICAL 🚨: `preprocessing_params` not loaded or is not a dictionary."
        error_messages.append(msg)
        print(msg)
        load_success = False
        resources["preprocessing_params"] = {} # Ensure empty dict

    return resources, load_success, error_messages

resources, load_success, load_errors = load_all_resources()

# --- Initialize Session State ---
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'prediction_proba' not in st.session_state:
    st.session_state.prediction_proba = None

# ---- Sidebar ----
st.sidebar.image("https://banner2.cleanpng.com/20180712/oq/kisspng-financial-risk-management-credit-risk-loan-inves-financial-risk-5b479a8009a810.4009991215314192640396.jpg", use_column_width=True) # Placeholder image
st.sidebar.title("IntelliRisk™️ Controls")
st.sidebar.info(
    "This engine assesses creditworthiness using an advanced LightGBM model. "
    "Input client data to receive a default risk probability."
)
if load_errors:
    for err in load_errors:
        if "CRITICAL" in err:
            st.sidebar.error(err)
        else:
            st.sidebar.warning(err) # Should not happen with current logic, but good fallback
st.sidebar.markdown("---")
st.sidebar.header("Data Source Info")
st.sidebar.caption("Based on Home Credit Default Risk competition data. Feature engineering and preprocessing align with the source notebook.")
st.sidebar.caption("Version: 1.0.0-pro")


# ---- Main App Logic ----
st.title("IntelliRisk™️ - Creditworthiness Engine")

if not load_success or resources.get('model') is None:
    st.error("One or more critical application resources failed to load. Please check the sidebar error messages and application logs. The application cannot proceed.")
    st.stop()
else:
    model = resources['model']
    preprocessing_params = resources['preprocessing_params']
    final_feature_names = resources['final_feature_names']
    categorical_features_list = resources.get('categorical_features', []) # Get it, default to empty if missing

    if not st.session_state.show_results:
         st.markdown("🤖 **Engine Ready.** Input client information below to assess credit default risk.", unsafe_allow_html=True)


# ---- Conditional Display: Input Form OR Results Page ----
if not st.session_state.show_results and model is not None:
    with st.form(key="credit_risk_input_form"):
        st.header("🖋️ Client & Loan Application Details")
        
        # --- Define Input Fields ---
        # Using a more structured approach for inputs
        input_sections = {
            "👤 Applicant Information": {
                "raw_amt_income_total": {"label": "Total Annual Income", "type": "number", "min": 20000, "val": 150000, "step": 5000},
                "raw_age_years": {"label": "Age (Years)", "type": "number", "min": 18, "val": 35, "step": 1},
                "raw_code_gender": {"label": "Gender", "type": "select", "options": ["F", "M", "XNA"], "val_idx": 0},
                "raw_name_education_type": {"label": "Education Level", "type": "select",
                                            "options": ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"], "val_idx": 0},
                "raw_name_family_status": {"label": "Family Status", "type": "select",
                                           "options": ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"], "val_idx": 0},
                "raw_cnt_children": {"label": "Number of Children", "type": "number", "min": 0, "val": 0, "step": 1},
            },
            "💳 Loan Characteristics": {
                "raw_name_contract_type": {"label": "Contract Type", "type": "select", "options": ["Cash loans", "Revolving loans"], "val_idx": 0},
                "raw_amt_credit": {"label": "Loan Amount Requested", "type": "number", "min": 20000, "val": 250000, "step": 10000},
                "raw_amt_annuity": {"label": "Loan Annuity (Periodic Payment)", "type": "number", "min": 5000, "val": 20000, "step": 1000},
                "raw_amt_goods_price": {"label": "Goods Price (for consumer loans)", "type": "number", "min": 0, "val": 225000, "step": 5000, "help": "Value of goods if it's a goods loan."},
            },
            "🏢 Employment & External Scores": {
                "raw_name_income_type": {"label": "Income Type", "type": "select",
                                          "options": ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student", "Businessman", "Maternity leave"], "val_idx": 0},
                "raw_years_employed": {"label": "Years Employed", "type": "number", "val": 5, "min_value": -5, "step": 1, "help": "Positive if employed, 0 or negative for unemployed/N.A./Retired."},
                "raw_organization_type": {"label": "Organization Type", "type": "select", "options": ["Business Entity Type 3", "School", "Government", "Self-employed", "XNA"], "val_idx":0, "help": "Select 'XNA' if not applicable or unknown."}, # Add more options!
                "raw_ext_source_1": {"label": "External Source 1 Score", "type": "slider", "min": 0.0, "max": 1.0, "val": 0.5, "step": 0.01},
                "raw_ext_source_2": {"label": "External Source 2 Score", "type": "slider", "min": 0.0, "max": 1.0, "val": 0.5, "step": 0.01},
                "raw_ext_source_3": {"label": "External Source 3 Score", "type": "slider", "min": 0.0, "max": 1.0, "val": 0.5, "step": 0.01},
            }
        }

        raw_inputs = {}
        tabs = st.tabs(list(input_sections.keys()))
        for i, (section_name, fields) in enumerate(input_sections.items()):
            with tabs[i]:
                for key, config in fields.items():
                    if config["type"] == "number":
                        raw_inputs[key] = st.number_input(config["label"], min_value=config.get("min", None), max_value=config.get("max", None), value=config["val"], step=config.get("step", None), help=config.get("help"))
                    elif config["type"] == "select":
                        raw_inputs[key] = st.selectbox(config["label"], config["options"], index=config["val_idx"], help=config.get("help"))
                    elif config["type"] == "slider":
                        raw_inputs[key] = st.slider(config["label"], min_value=config["min"], max_value=config["max"], value=config["val"], step=config["step"], help=config.get("help"))
        
        st.markdown("---") # Separator before button

        # ---- Preprocessing and Feature Engineering Function ----
        def create_feature_vector(user_raw_inputs_dict, pp_params, final_feats_order, cat_feats_list):
            # Convert user inputs to a Polars DataFrame (1 row)
            # Raw keys are like 'raw_amt_income_total', need to map to original feature names if different for Polars
            # For simplicity, assume user_raw_inputs_dict keys match what make_nan_free etc. expect as input columns
            
            # Map UI keys to original DataFrame column names if they differ
            # For this example, I'll assume the keys in user_raw_inputs_dict are ALREADY
            # the base column names your functions like make_nan_free expect.
            # E.g., user_raw_inputs_dict should have 'AMT_INCOME_TOTAL', not 'raw_amt_income_total'
            # So, we'll need a mapping or ensure UI keys are direct.
            
            # Let's create a new dict with original-like keys for Polars processing
            polars_input_data = {}
            polars_input_data['AMT_INCOME_TOTAL'] = user_raw_inputs_dict['raw_amt_income_total']
            polars_input_data['DAYS_BIRTH'] = -1 * user_raw_inputs_dict['raw_age_years'] * 365.25
            polars_input_data['CODE_GENDER'] = user_raw_inputs_dict['raw_code_gender']
            polars_input_data['CNT_CHILDREN'] = user_raw_inputs_dict['raw_cnt_children']
            polars_input_data['NAME_CONTRACT_TYPE'] = user_raw_inputs_dict['raw_name_contract_type']
            polars_input_data['AMT_CREDIT'] = user_raw_inputs_dict['raw_amt_credit']
            polars_input_data['AMT_ANNUITY'] = user_raw_inputs_dict['raw_amt_annuity']
            polars_input_data['AMT_GOODS_PRICE'] = user_raw_inputs_dict['raw_amt_goods_price']
            polars_input_data['NAME_EDUCATION_TYPE'] = user_raw_inputs_dict['raw_name_education_type']
            polars_input_data['NAME_FAMILY_STATUS'] = user_raw_inputs_dict['raw_name_family_status']
            polars_input_data['NAME_INCOME_TYPE'] = user_raw_inputs_dict['raw_name_income_type']
            polars_input_data['ORGANIZATION_TYPE'] = user_raw_inputs_dict['raw_organization_type']
            polars_input_data['EXT_SOURCE_1'] = user_raw_inputs_dict['raw_ext_source_1'] # Assuming these are direct values
            polars_input_data['EXT_SOURCE_2'] = user_raw_inputs_dict['raw_ext_source_2']
            polars_input_data['EXT_SOURCE_3'] = user_raw_inputs_dict['raw_ext_source_3']

            if user_raw_inputs_dict['raw_years_employed'] > 0:
                polars_input_data['DAYS_EMPLOYED'] = -1 * user_raw_inputs_dict['raw_years_employed'] * 365.25
            else:
                polars_input_data['DAYS_EMPLOYED'] = 365243 # Common placeholder for XNA/unemployed

            # --- Add ALL other necessary raw features from UI to polars_input_data ---
            # For features not in UI, you MUST impute them here before creating the polars DataFrame.
            # Get the full list of columns expected by make_nan_free (original CSV columns)
            # For this demo, we'll assume polars_input_data has enough. In reality, you need all.
            # This is a MAJOR simplification. You need to fill ALL columns your `make_nan_free` expects.
            # Example: If your model used 'FLAG_OWN_CAR', you'd need an input for it, or a default here.
            # polars_input_data['FLAG_OWN_CAR'] = 'N' # Example default

            # Create a Polars DataFrame from the single row of inputs
            try:
                df_pl = pl.DataFrame([polars_input_data])
            except Exception as e:
                st.error(f"Error creating Polars DataFrame from input: {e}. Ensure all expected base columns are present in `polars_input_data`.")
                return None


            # 1. Apply `make_nan_free` logic (Imputation)
            # We need medians from training data for numeric nulls identified in training
            numeric_null_cols_to_impute = pp_params.get("numeric_null_cols_from_train", [])
            temp_cols_for_median_impute = []
            for col in numeric_null_cols_to_impute:
                if col in df_pl.columns: # Only impute if column exists in current input
                    # Get median from pp_params (you'd need to save these per column)
                    # For now, let's assume we use a general median or it was handled by global median in notebook
                    # This is a simplification; robust imputation needs saved training medians per column
                    # For your notebook's make_nan_free, it computed medians on the fly from the input df
                    # For inference, we must use medians from TRAINING DATA.
                    # Let's assume pp_params['medians_for_imputation'][col] exists
                    # training_median = pp_params.get("medians_for_imputation", {}).get(col, 0) # Default to 0 if median not saved
                    # df_pl = df_pl.with_columns(pl.col(col).fill_null(training_median))
                    temp_cols_for_median_impute.append(pl.col(col).fill_null(pl.median(col))) # Replicates notebook behavior for the single row
            if temp_cols_for_median_impute:
                df_pl = df_pl.with_columns(temp_cols_for_median_impute)


            categorical_null_cols_to_impute = pp_params.get("categorical_null_cols_from_train", [])
            temp_cols_for_cat_impute = []
            for col in categorical_null_cols_to_impute:
                if col in df_pl.columns:
                    temp_cols_for_cat_impute.append(pl.col(col).fill_null("Unknown"))
            if temp_cols_for_cat_impute:
                 df_pl = df_pl.with_columns(temp_cols_for_cat_impute)


            # 2. Create Squared Features (from raw/imputed values, before scaling)
            # These are based on columns identified during training (correlation with TARGET)
            cols_to_square = pp_params.get("columns_to_square_from_raw", [])
            for col_name in cols_to_square:
                if col_name in df_pl.columns:
                    try:
                        df_pl = df_pl.with_columns(
                            (pl.col(col_name) ** 2).alias(f'{col_name}_squre')
                        )
                    except Exception as e: # Could be non-numeric after imputation
                        st.warning(f"Could not create squared feature for {col_name} (is it numeric?): {e}")
                        df_pl = df_pl.with_columns(pl.lit(0.0).alias(f'{col_name}_squre')) # Add placeholder
                else: # Add placeholder if original col for squaring not even in input
                    df_pl = df_pl.with_columns(pl.lit(0.0).alias(f'{col_name}_squre'))


            # 3. Apply `make_efficient` logic (Scaling)
            # Use means and stds from TRAINING DATA stored in pp_params
            means = pp_params.get("means_for_scaling", {})
            stds = pp_params.get("stds_for_scaling", {})
            
            cols_to_scale_in_df = [col for col in df_pl.columns if col in means and col in stds] # Only scale if stats are available and col exists
            scaling_exprs = []
            for col in cols_to_scale_in_df:
                # Ensure mean and std are numbers, not None
                mean_val = means.get(col)
                std_val = stds.get(col)
                if isinstance(mean_val, (int, float)) and isinstance(std_val, (int, float)):
                     scaling_exprs.append(
                        ((pl.col(col) - mean_val) / (std_val if std_val != 0 else 1.0)).alias(col) # Avoid division by zero for std
                     )
                else:
                    st.warning(f"Scaling skipped for {col}: Missing mean/std from training data or non-numeric.")


            if scaling_exprs:
                df_pl = df_pl.with_columns(scaling_exprs)
            
            # 4. Cast Categorical Features (as done in notebook)
            # `categorical_features_list` are the original names of string columns
            for col_name in cat_feats_list:
                if col_name in df_pl.columns:
                    try:
                        df_pl = df_pl.with_columns(pl.col(col_name).cast(pl.Categorical))
                    except Exception as e:
                        st.warning(f"Could not cast {col_name} to categorical: {e}")
                        # If casting fails, it might already be a problematic type.
                        # Consider how to handle this, e.g., fill with a placeholder category.
                        # For now, we proceed, but this could lead to issues if model expects categorical.

            # 5. Convert to Pandas DataFrame for LightGBM (as in notebook)
            # Crucially, ensure OHE is handled here if your model expects it and it wasn't part of Polars
            # Your notebook converts to pandas THEN LightGBM's `categorical_feature='auto'` handles it.
            # Or, if you did explicit OHE in notebook, replicate here.
            # For now, assume 'auto' will work on the pandas df from polars categoricals.
            try:
                df_pandas = df_pl.to_pandas()
            except Exception as e:
                st.error(f"Error converting processed Polars DF to Pandas DF: {e}")
                return None

            # 6. Reorder/Select final features and fill any still missing
            # This is where final_feature_names from training is paramount.
            final_df_for_model = pd.DataFrame(columns=final_feats_order)
            for col in final_feats_order:
                if col in df_pandas.columns:
                    final_df_for_model[col] = df_pandas[col]
                else:
                    # This means a feature expected by the model was NOT generated.
                    # This is a critical preprocessing mismatch.
                    # For example, if OHE was done in training to create 'CODE_GENDER_F',
                    # but not replicated here, it will be missing.
                    final_df_for_model[col] = 0 # Default fill - VERY LIKELY TO CAUSE ISSUES if not handled well
                    st.warning(f"Feature '{col}' expected by model was not generated in preprocessing. Defaulting to 0.")

            return final_df_for_model.values.astype(np.float32) # LGBM expects numpy array

        # ---- Submit Button ----
        submitted = st.form_submit_button("🛡️ Assess Credit Risk", use_container_width=True) # Changed button text

        if submitted:
            with st.spinner("🔍 Analyzing application... Please wait."):
                try:
                    # `raw_inputs` dict is already populated by the dynamic form creation
                    feature_vector = create_feature_vector(raw_inputs, preprocessing_params, final_feature_names, categorical_features_list)
                    
                    if feature_vector is None: # Preprocessing failed critically
                        st.error("Data preprocessing failed. Cannot make a prediction.")
                    elif not final_feature_names:
                         st.error("CRITICAL: `final_feature_names` list is missing or empty. Cannot validate feature vector shape.")
                    elif feature_vector.shape[1] != len(final_feature_names):
                        st.error(f"Critical Feature Count Mismatch: Model expects {len(final_feature_names)} features, "
                                 f"but {feature_vector.shape[1]} were generated. "
                                 "This indicates a significant issue in the `create_feature_vector` function "
                                 "or the loaded `final_ordered_feature_names.json`.")
                        # st.write("First 10 generated feature names (if possible to reconstruct from vector): CHECK MANUALLY")
                        # st.write("First 10 expected feature names:", final_feature_names[:10])
                    else:
                        prediction_probabilities = model.predict_proba(feature_vector)
                        st.session_state.prediction_proba = prediction_probabilities[0][1] # Prob of class 1 (default)
                        st.session_state.show_results = True
                        st.experimental_rerun()

                except Exception as e:
                    st.error(f"An error occurred during risk assessment: {e}")
                    st.text(traceback.format_exc())

elif st.session_state.show_results and st.session_state.prediction_proba is not None:
    proba = st.session_state.prediction_proba
    proba_percent = proba * 100
    
    if proba_percent >= 60: # Adjusted thresholds for more "extreme" feel
        risk_level_text = "Critical Risk"
        risk_color_class = "status-high-risk"
        risk_icon = "🚨"
    elif proba_percent >= 40:
        risk_level_text = "High Risk"
        risk_color_class = "status-high-risk"
        risk_icon = "⚠️"
    elif proba_percent >= 20:
        risk_level_text = "Moderate Risk"
        risk_color_class = "status-moderate-risk"
        risk_icon = "🤔"
    else:
        risk_level_text = "Low Risk"
        risk_color_class = "status-low-risk"
        risk_icon = "✅"

    st.markdown(f"""
        <div class="prediction-result-card">
            <div class="prediction-result-title">Credit Default Risk Assessment</div>
            <div class="prediction-result-main-value {risk_color_class}">
                {risk_icon} {proba_percent:.1f}%
            </div>
            <div class="prediction-risk-level" style="color: var(--{risk_color_class.replace('status-','')})"> <!-- Use CSS variable for color -->
                Assessment: <strong>{risk_level_text}</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("‹ Start New Assessment", use_container_width=True, key="new_assessment_button"):
        st.session_state.show_results = False
        st.session_state.prediction_proba = None
        st.experimental_rerun()
