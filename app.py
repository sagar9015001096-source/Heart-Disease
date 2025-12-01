import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# --- 1. Load Data and Train Model Function ---

@st.cache_resource
def load_and_train_model(csv_file):
    """Loads data, prepares the model pipeline, and trains the model."""
    
    # Load the dataset
    df = pd.read_csv(csv_file)
    
    # Data Cleaning & Target Binarization
    df_clean = df.copy()
    df_clean = df_clean.drop(['id', 'dataset'], axis=1)

    # Binarize the target variable 'num': 0 (No Disease) vs 1 (Disease)
    df_clean['target'] = df_clean['num'].apply(lambda x: 0 if x == 0 else 1)
    df_clean = df_clean.drop('num', axis=1)

    # --- Map numeric codes to labels to match UI ---

    # Sex: 0 = female, 1 = male
    sex_map = {0: "Female", 1: "Male"}
    df_clean['sex'] = df_clean['sex'].map(sex_map).fillna(df_clean['sex'])

    # Chest pain type (cp)
    cp_map = {
        1: 'typical angina',
        2: 'atypical angina',
        3: 'non-anginal',
        4: 'asymptomatic'
    }
    df_clean['cp'] = df_clean['cp'].map(cp_map).fillna(df_clean['cp'])

    # Resting ECG (restecg)
    restecg_map = {
        0: 'normal',
        1: 'st-t abnormality',
        2: 'lv hypertrophy'
    }
    df_clean['restecg'] = df_clean['restecg'].map(restecg_map).fillna(df_clean['restecg'])

    # Slope
    slope_map = {
        1: 'upsloping',
        2: 'flat',
        3: 'downsloping'
    }
    df_clean['slope'] = df_clean['slope'].map(slope_map).fillna(df_clean['slope'])

    # Thallium (thal)
    thal_map = {
        3: 'normal',
        6: 'fixed defect',
        7: 'reversable defect'
    }
    df_clean['thal'] = df_clean['thal'].map(thal_map).fillna(df_clean['thal'])

    # Fasting blood sugar & exercise induced angina: 0/1 -> "False"/"True"
    fbs_map = {0: 'False', 1: 'True'}
    exang_map = {0: 'False', 1: 'True'}
    df_clean['fbs'] = df_clean['fbs'].map(fbs_map).astype(str)
    df_clean['exang'] = df_clean['exang'].map(exang_map).astype(str)
    
    # Separate features (X) and target (y)
    X = df_clean.drop('target', axis=1)
    y = df_clean['target']

    # Define Feature Types
    numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

    # Preprocessor for numerical features: impute with median, then scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessor for categorical features: impute with most frequent, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a ColumnTransformer to apply transformations to the correct columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create the full pipeline (Preprocessor + Model)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Simple evaluation on the held-out test set
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Return model + feature columns + accuracy + cleaned data for dashboard
    return model_pipeline, X.columns.tolist(), accuracy, df_clean


# --- 2. Basic Streamlit Config & Session State ---

st.set_page_config(page_title="Heart Disease Prediction App", layout="centered")

# Load model and data
try:
    model_pipeline, feature_cols, model_accuracy, data = load_and_train_model("heart_disease_uci.csv")
except Exception as e:
    st.error(f"Error loading and training model: {e}")
    st.stop()

# Session state for prediction history
if "history" not in st.session_state:
    st.session_state["history"] = []


# --- 3. Helper: Risk label ---

def get_risk_label(prob_disease: float) -> str:
    """Return a text label for risk category based on probability of disease."""
    if prob_disease < 0.25:
        return "Low"
    elif prob_disease < 0.50:
        return "Moderate"
    elif prob_disease < 0.75:
        return "High"
    else:
        return "Very High"


# --- 4. Page Functions ---

def show_home_page():
    st.title("💖 Heart Disease Prediction System")
    st.markdown("### Welcome to the Heart Disease ML Web App")

    st.write(
        """
This project is a **machine learning–based web application** that predicts the 
likelihood of heart disease using clinical input parameters.

Use the sidebar to navigate between:
- 🏠 **Home** – You are here
- 🩺 **Prediction** – Enter patient data and get risk estimation
- 📊 **Dashboard** – Explore dataset statistics and visualizations
- ℹ️ **About** – Learn about the project and dataset
        """
    )

    # Some quick stats from data
    total_records = len(data)
    disease_cases = int((data["target"] == 1).sum())
    no_disease_cases = int((data["target"] == 0).sum())
    disease_pct = disease_cases / total_records * 100

    st.markdown("### 📈 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", total_records)
    col2.metric("Heart Disease Cases", disease_cases, f"{disease_pct:.1f}%")
    col3.metric("Model Accuracy", f"{model_accuracy * 100:.1f}%")

    st.markdown("---")
    st.markdown("### 🔍 Sample of the Dataset")
    st.dataframe(data.head(), use_container_width=True)


def show_prediction_page():
    st.title("🩺 Heart Disease Risk Prediction")

    st.write(
        """
Provide the patient's clinical details below and click 
**'Predict Heart Disease Risk'** to get an estimated risk.
        """
    )

    with st.form("prediction_form"):
        st.header("Patient Clinical Data")
        
        # --- Row 1: Age and Sex ---
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 100, 50, help="Age of the patient in years.")
        with col2:
            sex = st.selectbox("Sex", ("Male", "Female"), help="Patient's biological sex.")

        # --- Row 2: Chest Pain and Resting BP ---
        col3, col4 = st.columns(2)
        with col3:
            cp = st.selectbox(
                "Chest Pain Type (cp)", 
                ('typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'),
                help="Type of chest pain experienced."
            )
        with col4:
            trestbps = st.number_input(
                "Resting Blood Pressure (trestbps)", 80, 200, 120, 
                help="Resting blood pressure in mm Hg."
            )

        # --- Row 3: Cholesterol and Fasting Blood Sugar ---
        col5, col6 = st.columns(2)
        with col5:
            chol = st.number_input(
                "Serum Cholesterol (chol)", 100, 600, 240, 
                help="Serum cholesterol in mg/dl."
            )
        with col6:
            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl (fbs)", 
                ('True', 'False'),
                help="Fasting Blood Sugar > 120 mg/dl (True/False)."
            )

        # --- Row 4: Resting ECG and Max Heart Rate ---
        col7, col8 = st.columns(2)
        with col7:
            restecg = st.selectbox(
                "Resting Electrocardiographic Results (restecg)", 
                ('lv hypertrophy', 'normal', 'st-t abnormality'),
                help="ECG results at rest."
            )
        with col8:
            thalch = st.number_input(
                "Maximum Heart Rate Achieved (thalch)", 70, 220, 150, 
                help="Max heart rate during the stress test."
            )

        # --- Row 5: Exercise Induced Angina and Oldpeak ---
        col9, col10 = st.columns(2)
        with col9:
            exang = st.selectbox(
                "Exercise Induced Angina (exang)", 
                ('True', 'False'),
                help="Presence of angina induced by exercise."
            )
        with col10:
            oldpeak = st.number_input(
                "ST Depression Induced by Exercise (oldpeak)",
                0.0, 6.2, 1.0, step=0.1, 
                help="ST depression relative to rest."
            )

        # --- Row 6: Slope, CA, and Thallium Scan ---
        col11, col12, col13 = st.columns(3)
        with col11:
            slope = st.selectbox(
                "Slope of Peak Exercise ST Segment (slope)", 
                ('upsloping', 'flat', 'downsloping'),
                help="The slope of the peak exercise ST segment."
            )
        with col12:
            ca = st.slider(
                "Number of Major Vessels Colored by Flouroscopy (ca)", 0, 3, 0, 
                help="Number of major vessels (0-3) colored by flouroscopy."
            )
        with col13:
            thal = st.selectbox(
                "Thallium Stress Test Result (thal)", 
                ('normal', 'fixed defect', 'reversable defect'),
                help="Result of the thallium stress test."
            )

        submitted = st.form_submit_button("🔍 Predict Heart Disease Risk")

    if submitted:
        # 1. Create a dictionary of all inputs
        input_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalch': thalch,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        
        # 2. Convert to DataFrame (preserving the feature order/columns used in training)
        input_df = pd.DataFrame([input_data], columns=feature_cols)
        
        # 3. Make Prediction
        try:
            prediction = model_pipeline.predict(input_df)[0]
            prediction_proba = model_pipeline.predict_proba(input_df)[0]

            prob_no_disease = float(prediction_proba[0])
            prob_disease = float(prediction_proba[1])
            risk_label = get_risk_label(prob_disease)
            risk_percent = prob_disease * 100
            
            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("**High Risk of Heart Disease (Predicted: 1)**")
                st.markdown(f"**Probability of Disease:** `{prob_disease:.2%}`")
                st.warning("Please consult a healthcare professional for further evaluation.")
            else:
                st.success("**Low Risk of Heart Disease (Predicted: 0)**")
                st.markdown(f"**Probability of No Disease:** `{prob_no_disease:.2%}`")
                st.info("The model suggests a low risk. This is not a substitute for medical advice.")

            # --- Risk level meter ---
            st.markdown("### 📉 Risk Level Overview")
            st.write(f"**Risk Category:** {risk_label} ({risk_percent:.1f}%)")
            st.progress(int(risk_percent))

            # --- Save prediction to session history ---
            result_row = input_data.copy()
            result_row.update({
                "predicted_label": int(prediction),
                "probability_no_disease": prob_no_disease,
                "probability_disease": prob_disease,
                "risk_level": risk_label
            })
            st.session_state["history"].append(result_row)

            # --- Download this single prediction as CSV ---
            single_result_df = pd.DataFrame([result_row])
            single_csv = single_result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download this prediction as CSV",
                single_csv,
                "single_heart_disease_prediction.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")


def show_dashboard_page():
    st.title("📊 Heart Disease Data Dashboard")

    st.write(
        """
Explore the dataset used to train the model.  
These visualizations can be used in your **final-year project report / PPT**.
        """
    )

    # Target distribution
    st.subheader("Target Distribution (0 = No Disease, 1 = Disease)")
    target_counts = data["target"].value_counts().sort_index()
    st.bar_chart(target_counts)

    # Average cholesterol by target
    st.subheader("Average Cholesterol by Heart Disease Status")
    chol_by_target = data.groupby("target")["chol"].mean()
    st.bar_chart(chol_by_target)

    # Age distribution
    st.subheader("Age Distribution")
    st.caption("Histogram-style view using value counts (grouped by age).")
    age_counts = data["age"].value_counts().sort_index()
    st.line_chart(age_counts)

    # Show prediction history if any
    if st.session_state["history"]:
        st.markdown("---")
        st.subheader("📁 Prediction History (this session)")
        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df, use_container_width=True)

        history_csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download full prediction history as CSV",
            history_csv,
            "heart_disease_prediction_history.csv",
            "text/csv"
        )


def show_about_page():
    st.title("ℹ️ About This Project")

    st.markdown(
        """
### 🔧 Project Overview
This web application is a **Heart Disease Prediction System** built as a 
**Machine Learning + Web (Streamlit) project**.

It:
- Loads a heart disease dataset
- Performs preprocessing (cleaning, encoding, scaling)
- Trains a **Random Forest Classifier**
- Provides a **web-based UI** to:
  - Input patient clinical data
  - Predict the probability of heart disease
  - Visualize dataset statistics

---

### 🧠 Machine Learning Details

- **Algorithm:** RandomForestClassifier  
- **Task:** Binary classification  
- **Target:** Presence (1) or absence (0) of heart disease  
- **Validation:** Train–test split (80%–20%)  

---

### ⚠️ Disclaimer

This project is intended **for educational and academic purposes only**.  
It **must not** be used as a replacement for professional medical advice, diagnosis, or treatment.

Always consult a **qualified healthcare provider** with any questions about medical conditions.
        """
    )


# --- 5. Sidebar Navigation ---

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Prediction", "Dashboard", "About"]
)

st.sidebar.markdown("---")
st.sidebar.metric("Model Accuracy", f"{model_accuracy * 100:.1f}%")

# --- 6. Render Selected Page ---

if page == "Home":
    show_home_page()
elif page == "Prediction":
    show_prediction_page()
elif page == "Dashboard":
    show_dashboard_page()
else:
    show_about_page()
