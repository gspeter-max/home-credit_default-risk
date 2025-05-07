import streamlit as st
import numpy as np
import pandas as pd
import joblib
import traceback
import json # For loading feature names list

# ---- Page Configuration ----
st.set_page_config(
    page_title="Home Credit Default Risk Predictor 🏦",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Same dark theme as before for consistency) ---
st.markdown("""
    <style>
        body, .stApp { color: #EAEAEA; background-color: #0E1117; }
        h1, h2, h3, h4, h5, h6 { color: #FFFFFF; font-weight: 600;}
        .css-1d391kg { background-color: #161A1F; border-right: 1px solid #30363F; }
        .css-1d391kg .stMarkdown p, .css-1d391kg .stMarkdown li { color: #C0C0C0; }
        div[data-testid="stWidgetLabel"] label p { color: #D0D0D0 !important; font-size: 0.95em !important; }
        .stMultiSelect div[data-baseweb="block"] > div:first-child { color: #D0D0D0 !important; }
        .stTextInput > div > div > input, .stNumberInput > div > div > input,
        .stSelectbox > div > div > div, .stMultiSelect > div > div > div > div {
            background-color: #20242A; color: #EAEAEA; border-radius: 0.3rem; border: 1px solid #3A3F4A;
        }
        .stButton>button {
            border-radius: 0.3rem; background-color: #0078D4; color: white;
            border: none; padding: 0.6em 1.2em; font-weight: 500;
            transition: background-color 0.2s ease-in-out;
        }
        .stButton>button:hover { background-color: #005A9E; }
        .stButton>button:active { background-color: #004C82; }
        .streamlit-expanderHeader { font-size: 1.1em; font-weight: 500; color: #B0B0B0; }
        .prediction-result-area {
            text-align: center; padding: 2em; margin: 2em auto; border: 1px solid #30363F;
            border-radius: 0.75rem; background: linear-gradient(145deg, #1A1C22, #20242A);
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.3); max-width: 600px;
        }
        .prediction-result-header {
            font-size: 1.5em; color: #A0A0A5; margin-bottom: 0.75em;
            text-transform: uppercase; letter-spacing: 1px;
        }
        .prediction-result-value { font-size: 3.8em; font-weight: 700; margin-bottom: 0.3em; line-height: 1.1; }
        .probability-default { color: #E74C3C; }
        .probability-no-default { color: #2ECC71; }
        .key-factors-container { margin-top: 1.5em; padding-top: 1em; border-top: 1px dashed #3A3F4A; }
        .key-factors-title { font-size: 1.1em; color: #A0A0A5; text-align: center; margin-bottom: 0.75em;}
        .factor-badge {
            display: inline-block; background-color: #30363F; color: #B0B0B0;
            padding: 0.3em 0.6em; border-radius: 1em; font-size: 0.9em; margin: 0.2em;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Define Paths for Model and Preprocessing Objects ----
# !!! UPDATE THESE PATHS IF YOUR FILENAMES ARE DIFFERENT !!!
MODEL_PATH = 'credit_risk.pkl'
SCALER_PATH = 'scaler.pkl'  # Example: You need to save your main scaler
ENCODER_GENDER_PATH = 'encoder_gender.pkl' # Example: Save your gender encoder
ENCODER_CONTRACT_PATH = 'encoder_contract_type.pkl' # Example
# ... add paths for ALL other scalers/encoders you used ...
FINAL_FEATURES_PATH = 'final_feature_names.json' # List of feature names in order

@st.cache_resource
def load_resources():
    resources = {'model': None, 'scaler': None, 'encoder_gender': None, 'encoder_contract_type': None, 'final_feature_names': []}
    error_messages = []
    
    try:
        resources['model'] = joblib.load(MODEL_PATH)
    except Exception as e:
        error_messages.append(f"❌ Error loading model ('{MODEL_PATH}'): {e}")

    try:
        resources['scaler'] = joblib.load(SCALER_PATH)
    except Exception: # Be more specific if you know the expected error (e.g., FileNotFoundError)
        error_messages.append(f"⚠️ Scaler ('{SCALER_PATH}') not found. Numerical inputs may not be processed correctly.")
        # For demo, you might create a dummy scaler that does nothing if you want the app to run
        # class DummyScaler:
        #     def transform(self, X): return X
        # resources['scaler'] = DummyScaler()


    try:
        resources['encoder_gender'] = joblib.load(ENCODER_GENDER_PATH)
    except Exception:
        error_messages.append(f"⚠️ Gender Encoder ('{ENCODER_GENDER_PATH}') not found. Gender may not be processed correctly.")

    try:
        resources['encoder_contract_type'] = joblib.load(ENCODER_CONTRACT_PATH)
    except Exception:
        error_messages.append(f"⚠️ Contract Type Encoder ('{ENCODER_CONTRACT_PATH}') not found. Contract type may not be processed correctly.")

    # ... Load ALL your other scalers and encoders here ...

    try:
        with open(FINAL_FEATURES_PATH, 'r') as f:
            resources['final_feature_names'] = json.load(f)
        if not resources['final_feature_names']: # Check if list is empty
             error_messages.append(f"⚠️ Feature names list ('{FINAL_FEATURES_PATH}') is empty or not loaded correctly.")
    except Exception:
        error_messages.append(f"⚠️ Feature names list ('{FINAL_FEATURES_PATH}') not found. Feature order and completeness critical.")
        # If final_feature_names.json is missing, the app CANNOT work correctly.
        # Forcing an error or providing a hardcoded (but likely incorrect) default for demo:
        # resources['final_feature_names'] = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', ...] # VERY BAD PRACTICE - MUST MATCH TRAINING

    return resources, error_messages

resources, load_errors = load_resources()
model = resources.get('model')
scaler = resources.get('scaler') # Will be None or DummyScaler if not loaded
encoder_gender = resources.get('encoder_gender')
encoder_contract_type = resources.get('encoder_contract_type')
final_feature_names = resources.get('final_feature_names', []) # Default to empty list if not found

# --- Initialize Session State ---
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'prediction_proba' not in st.session_state:
    st.session_state.prediction_proba = None

# ---- Sidebar ----
st.sidebar.title("ℹ️ App Information")
st.sidebar.info(
    "Predicts the probability of a Home Credit client defaulting on their loan. "
    "Ensure all preprocessing objects (scalers, encoders) are correctly loaded for accurate results."
)
if load_errors:
    for err in load_errors:
        if "❌" in err:
            st.sidebar.error(err)
        else:
            st.sidebar.warning(err)
st.sidebar.markdown("---")
st.sidebar.caption("Home Credit Default Risk v0.1")

# ---- Main App Logic ----
st.title("🏦 Home Credit Default Risk Predictor")

if model is None: # If model itself failed to load, stop.
    st.error("CRITICAL: Main prediction model could not be loaded. Application cannot proceed.")
    st.stop()
else:
    if not st.session_state.show_results:
         st.markdown("🤖 **Model loaded.** Please fill in the client's details below. "
                     "Note: This UI uses a simplified set of inputs. Many other features are "
                     "derived or use default values for this demonstration.")

# ---- Conditional Display: Input Form OR Results Page ----
if not st.session_state.show_results and model is not None:
    with st.form(key="default_risk_input_form"):
        st.header("📝 Client & Loan Information (Selected Features)")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            with st.expander("👤 Basic Demographics", expanded=True):
                raw_amt_income_total = st.number_input("Total Annual Income", min_value=25000, value=135000, step=5000)
                raw_age_years = st.number_input("Age (Years)", min_value=18, max_value=70, value=35, step=1)
                raw_code_gender = st.selectbox("Gender", ["F", "M", "XNA"], index=0, help="F: Female, M: Male, XNA: Other/Not Available")
                # Add NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS etc. as needed from your important features
                raw_cnt_children = st.number_input("Number of Children", min_value=0, max_value=20, value=0, step=1)


        with col2:
            with st.expander("💳 Loan Details", expanded=True):
                raw_name_contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"], index=0)
                raw_amt_credit = st.number_input("Loan Amount (Credit Requested)", min_value=25000, value=250000, step=10000)
                raw_amt_annuity = st.number_input("Loan Annuity (Periodic Payment)", min_value=5000, value=20000, step=1000)
                raw_amt_goods_price = st.number_input("Goods Price (for consumer loans)", min_value=0, value=225000, step=5000)


        with col3:
            with st.expander("🏢 Employment & Other Key Factors", expanded=True):
                raw_years_employed = st.number_input("Years Employed (0 if unemployed/retired)", value=5, min_value=-5, step=1, help="Enter positive for employed, 0 or negative for unemployed/N.A.")
                raw_ext_source_2 = st.slider("External Source 2 Score (Normalized)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, help="A key anonymized score (0-1).")
                raw_ext_source_3 = st.slider("External Source 3 Score (Normalized)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, help="Another key anonymized score (0-1).")
                # You'd add more important features like ORGANIZATION_TYPE, OCCUPATION_TYPE etc. here

        # ---- Preprocessing and Feature Engineering Function ----
        def create_feature_vector(raw_inputs_dict):
            # THIS IS THE MOST CRITICAL FUNCTION. IT MUST EXACTLY REPLICATE YOUR TRAINING PREPROCESSING.
            # This is a SKELETON. You need to fill this with YOUR logic.

            # Create a Pandas Series or DataFrame from raw inputs for easier manipulation if needed
            df_input = pd.DataFrame([raw_inputs_dict])
            df_processed = pd.DataFrame() # To build up processed features

            # 1. Convert user-friendly inputs to model-expected format
            df_processed['DAYS_BIRTH'] = -1 * df_input['raw_age_years'] * 365.25
            
            if df_input['raw_years_employed'].iloc[0] > 0:
                df_processed['DAYS_EMPLOYED'] = -1 * df_input['raw_years_employed'].iloc[0] * 365.25
            else: # Handle unemployed/XNA as per your training (e.g., 365243 or NaN then impute)
                df_processed['DAYS_EMPLOYED'] = 365243 # Placeholder for XNA/unemployed, common in this dataset


            # 2. Handle Categorical Features (using loaded encoders or dummy creation)
            # Example: Gender (if you used one-hot encoding like CODE_GENDER_F, CODE_GENDER_M)
            # This assumes your final features include these explicit columns
            if encoder_gender: # If a gender encoder was loaded
                 # Example: gender_encoded = encoder_gender.transform(df_input[['raw_code_gender']])
                 # Then assign to appropriate columns in df_processed
                 pass # Replace with your actual encoding logic
            else: # Manual dummy for demo
                df_processed['CODE_GENDER_F'] = (df_input['raw_code_gender'] == 'F').astype(int)
                df_processed['CODE_GENDER_M'] = (df_input['raw_code_gender'] == 'M').astype(int)
                # If XNA was a category, add df_processed['CODE_GENDER_XNA'] = (df_input['raw_code_gender'] == 'XNA').astype(int)

            # Example: Contract Type
            if encoder_contract_type:
                pass # Replace with your actual encoding logic
            else: # Manual dummy for demo
                df_processed['NAME_CONTRACT_TYPE_Cash loans'] = (df_input['raw_name_contract_type'] == 'Cash loans').astype(int)
                df_processed['NAME_CONTRACT_TYPE_Revolving loans'] = (df_input['raw_name_contract_type'] == 'Revolving loans').astype(int)
            
            # ... ADD ALL YOUR OTHER CATEGORICAL ENCODING HERE ...
            # (e.g., NAME_EDUCATION_TYPE, ORGANIZATION_TYPE, etc.)

            # 3. Directly pass through numerical features that might be scaled later (or are already scaled if user inputs that)
            df_processed['AMT_INCOME_TOTAL'] = df_input['raw_amt_income_total']
            df_processed['AMT_CREDIT'] = df_input['raw_amt_credit']
            df_processed['AMT_ANNUITY'] = df_input['raw_amt_annuity']
            df_processed['AMT_GOODS_PRICE'] = df_input['raw_amt_goods_price']
            df_processed['CNT_CHILDREN'] = df_input['raw_cnt_children']
            df_processed['EXT_SOURCE_2'] = df_input['raw_ext_source_2'] # Assuming user inputs this scaled value
            df_processed['EXT_SOURCE_3'] = df_input['raw_ext_source_3'] # Assuming user inputs this scaled value


            # 4. Create Engineered Features (MUST MATCH TRAINING)
            # Example: df_processed['CREDIT_TO_INCOME_RATIO'] = df_processed['AMT_CREDIT'] / (df_processed['AMT_INCOME_TOTAL'] + 1e-6)
            # Example from your previous data snippet (you need to confirm how DAYS_BIRTH_squre was calculated - before/after scaling?)
            # Assuming DAYS_BIRTH is already negative days.
            # For this demo, let's assume it's squared value of DAYS_BIRTH.
            # You need to be precise if scaling was involved before squaring.
            df_processed['DAYS_BIRTH_squre'] = df_processed['DAYS_BIRTH'] ** 2 # EXAMPLE - VERIFY YOUR FORMULA


            # ... ADD ALL YOUR OTHER ENGINEERED FEATURES HERE ...


            # 5. Scaling (Numerical Features) - Placeholder, you need your actual logic
            # This is a very simplified example. You likely scaled many columns.
            numerical_cols_to_scale = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_BIRTH', 'DAYS_EMPLOYED'] # ADD ALL RELEVANT COLS
            if scaler:
                try:
                    # Make sure df_processed only contains columns the scaler expects, in the correct order, if scaler is specific
                    # If scaler was fitted on a DataFrame with specific columns:
                    # temp_df_for_scaling = df_processed[scaler.feature_names_in_] # If scaler has this attribute
                    # scaled_data = scaler.transform(temp_df_for_scaling)
                    # df_scaled = pd.DataFrame(scaled_data, columns=scaler.feature_names_in_, index=df_processed.index)
                    # df_processed.update(df_scaled)

                    # For a simple scaler on a subset of columns:
                    for col in numerical_cols_to_scale:
                        if col in df_processed.columns: # Ensure column exists before trying to scale
                            df_processed[col] = scaler.transform(df_processed[[col]]) # Scale one by one if scaler is simple
                        else:
                             st.sidebar.warning(f"Column '{col}' for scaling not found in processed features.")
                except Exception as e:
                    st.warning(f"Error applying scaling: {e}. Some numerical features might be unscaled.")
            else:
                st.warning("Scaler not loaded. Numerical features are unscaled.")

            # 6. Assemble final feature vector in the correct order using final_feature_names
            final_vector_data = {}
            for col in final_feature_names:
                if col in df_processed.columns:
                    final_vector_data[col] = df_processed[col].iloc[0]
                else:
                    final_vector_data[col] = 0 # Default for missing columns (YOU MUST HAVE A BETTER STRATEGY)
            
            final_df = pd.DataFrame([final_vector_data], columns=final_feature_names) # Ensure correct order and all columns
            return final_df.values.astype(np.float32) # Model expects numpy array


        # ---- Submit Button ----
        submitted = st.form_submit_button("🔍 Predict Default Risk", use_container_width=True)

        if submitted:
            # Collect all raw inputs from the form widgets
            raw_inputs = {
                'raw_amt_income_total': raw_amt_income_total,
                'raw_age_years': raw_age_years,
                'raw_code_gender': raw_code_gender,
                'raw_cnt_children': raw_cnt_children,
                'raw_name_contract_type': raw_name_contract_type,
                'raw_amt_credit': raw_amt_credit,
                'raw_amt_annuity': raw_amt_annuity,
                'raw_amt_goods_price': raw_amt_goods_price,
                'raw_years_employed': raw_years_employed,
                'raw_ext_source_2': raw_ext_source_2,
                'raw_ext_source_3': raw_ext_source_3
                # ... ADD ALL OTHER WIDGET VALUES HERE ...
            }
            
            with st.spinner("⚙️ Analyzing data and predicting..."):
                try:
                    feature_vector = create_feature_vector(raw_inputs)
                    
                    EXPECTED_NUM_FEATURES = len(final_feature_names)
                    if not final_feature_names:
                         st.error("CRITICAL: `final_feature_names` list is missing or empty. Cannot validate feature vector.")
                    elif feature_vector.shape[1] != EXPECTED_NUM_FEATURES:
                        st.error(f"Critical Feature Mismatch: Model expects {EXPECTED_NUM_FEATURES} features, "
                                 f"but {feature_vector.shape[1]} were generated from `create_feature_vector`. "
                                 "Ensure `final_feature_names.json` is loaded and `create_feature_vector` is complete.")
                        # st.write("Generated (first 10):", feature_vector[0, :10] if feature_vector.size > 0 else "Empty vector")
                        # st.write("Expected (first 10):", final_feature_names[:10])
                    else:
                        prediction_probabilities = model.predict_proba(feature_vector)
                        st.session_state.prediction_proba = prediction_probabilities[0][1] # Prob of class 1 (default)
                        st.session_state.show_results = True
                        st.experimental_rerun()

                except Exception as e:
                    st.error(f"An error occurred during preprocessing or prediction: {e}")
                    st.text(traceback.format_exc())

elif st.session_state.show_results and st.session_state.prediction_proba is not None:
    proba = st.session_state.prediction_proba
    risk_level_text = "High Risk of Default" if proba >= 0.5 else ("Moderate Risk" if proba >= 0.25 else "Low Risk of Default")
    risk_color_class = "probability-default" if proba >= 0.25 else "probability-no-default" # Red if proba >= 0.25

    st.markdown(f"""
        <div class="prediction-result-area">
            <div class="prediction-result-header">Predicted Probability of Default</div>
            <div class="prediction-result-value {risk_color_class}">
                {proba*100:.1f}%
            </div>
            <p style="font-size: 1.2em; color: #B0B0B0;">This applicant is considered: <strong>{risk_level_text}</strong></p>
        </div>
    """, unsafe_allow_html=True)

    # You could add a "Key Factors" teaser here if you implement basic logic for it

    if st.button("‹ Make Another Prediction", use_container_width=True):
        st.session_state.show_results = False
        st.session_state.prediction_proba = None
        st.experimental_rerun()
