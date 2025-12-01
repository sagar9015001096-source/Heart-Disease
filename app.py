import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -------------------- MODEL LOADING --------------------

@st.cache_resource
def load_and_train_model(csv_file):
    df = pd.read_csv(csv_file)
    df_clean = df.copy().drop(['id', 'dataset'], axis=1)
    df_clean['target'] = df_clean['num'].apply(lambda x: 0 if x == 0 else 1)
    df_clean = df_clean.drop('num', axis=1)

    sex_map = {0: "Female", 1: "Male"}
    cp_map = {1: 'typical angina', 2: 'atypical angina', 3: 'non-anginal', 4: 'asymptomatic'}
    restecg_map = {0: 'normal', 1: 'st-t abnormality', 2: 'lv hypertrophy'}
    slope_map = {1: 'upsloping', 2: 'flat', 3: 'downsloping'}
    thal_map = {3: 'normal', 6: 'fixed defect', 7: 'reversable defect'}
    fbs_map = {0: 'False', 1: 'True'}
    exang_map = {0: 'False', 1: 'True'}

    df_clean['sex'] = df_clean['sex'].map(sex_map).fillna(df_clean['sex'])
    df_clean['cp'] = df_clean['cp'].map(cp_map).fillna(df_clean['cp'])
    df_clean['restecg'] = df_clean['restecg'].map(restecg_map).fillna(df_clean['restecg'])
    df_clean['slope'] = df_clean['slope'].map(slope_map).fillna(df_clean['slope'])
    df_clean['thal'] = df_clean['thal'].map(thal_map).fillna(df_clean['thal'])
    df_clean['fbs'] = df_clean['fbs'].map(fbs_map).astype(str)
    df_clean['exang'] = df_clean['exang'].map(exang_map).astype(str)

    X = df_clean.drop('target', axis=1)
    y = df_clean['target']

    num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                      ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, X.columns.tolist(), accuracy, df_clean


st.set_page_config(page_title="Heart Disease Prediction App", layout="wide")

try:
    model, feature_cols, model_accuracy, data = load_and_train_model("heart_disease_uci.csv")
except:
    st.error("CSV or script error — check files on GitHub")
    st.stop()


# -------------------- NAVIGATION STATE --------------------

if "page" not in st.session_state:
    st.session_state.page = "Home"

def navigate(page):
    st.session_state.page = page


# -------------------- CSS NAVBAR --------------------

navbar_css = """
<style>
.navbar {
    display: flex;
    justify-content: center;
    background-color: #003B73;
    padding: 12px;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    z-index: 100;
}
.navbar a {
    color: white;
    text-decoration: none;
    margin: 0 22px;
    font-size: 18px;
    font-weight: 600;
    padding: 6px;
}
.navbar a:hover {
    color: #FFDD00;
    cursor: pointer;
}
.active {
    border-bottom: 2px solid #FFDD00;
}
body { padding-top: 70px; }
</style>
"""

st.markdown(navbar_css, unsafe_allow_html=True)

nav = f"""
<div class="navbar">
  <a class="{ 'active' if st.session_state.page=='Home' else '' }" onclick="fetch('/?page=Home')">Home</a>
  <a class="{ 'active' if st.session_state.page=='Prediction' else '' }" onclick="fetch('/?page=Prediction')">Prediction</a>
  <a class="{ 'active' if st.session_state.page=='Dashboard' else '' }" onclick="fetch('/?page=Dashboard')">Dashboard</a>
  <a class="{ 'active' if st.session_state.page=='About' else '' }" onclick="fetch('/?page=About')">About</a>
</div>
"""
st.markdown(nav, unsafe_allow_html=True)


# 🔄 Detect navbar clicks
query_params = st.experimental_get_query_params()
if "page" in query_params:
    st.session_state.page = query_params["page"][0]


# -------------------- PAGE CONTENT --------------------

def home_page():
    st.title("🏠 Heart Disease Prediction System")
    st.write("""
This ML-powered web application predicts the probability of heart disease 
based on patient clinical data. Use the navigation bar above to explore the app.
""")
    total = len(data)
    disease = int((data["target"] == 1).sum())
    st.metric("🧪 Model Accuracy", f"{model_accuracy*100:.1f}%")
    st.metric("📌 Dataset Size", total)
    st.metric("❤️ Disease Cases", disease)


def prediction_page():
    st.title("🩺 Heart Disease Risk Prediction")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1: age = st.slider("Age", 18, 100, 50)
        with col2: sex = st.selectbox("Sex", ("Male", "Female"))

        col3, col4 = st.columns(2)
        with col3:
            cp = st.selectbox("Chest Pain Type", ('typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'))
        with col4:
            trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)

        col5, col6 = st.columns(2)
        with col5: chol = st.number_input("Cholesterol", 100, 600, 240)
        with col6: fbs = st.selectbox("Fasting Blood Sugar > 120?", ('True', 'False'))

        col7, col8 = st.columns(2)
        with col7:
            restecg = st.selectbox("Resting ECG", ('lv hypertrophy', 'normal', 'st-t abnormality'))
        with col8:
            thalch = st.number_input("Max Heart Rate", 70, 220, 150)

        col9, col10 = st.columns(2)
        with col9: exang = st.selectbox("Exercise Induced Angina", ('True', 'False'))
        with col10: oldpeak = st.number_input("Oldpeak", 0.0, 6.2, 1.0)

        col11, col12, col13 = st.columns(3)
        with col11: slope = st.selectbox("Slope", ('upsloping', 'flat', 'downsloping'))
        with col12: ca = st.slider("Major Vessels", 0, 3, 0)
        with col13: thal = st.selectbox("Thallium Test", ('normal', 'fixed defect', 'reversable defect'))

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
                      'restecg': restecg, 'thalch': thalch, 'exang': exang, 'oldpeak': oldpeak,
                      'slope': slope, 'ca': ca, 'thal': thal}
        df = pd.DataFrame([input_data], columns=feature_cols)
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        st.subheader("Prediction Result")
        if pred == 1:
            st.error(f"High Risk ({prob:.2%} probability)")
        else:
            st.success(f"Low Risk ({1-prob:.2%} probability)")

def dashboard_page():
    st.title("📊 Dashboard")
    st.bar_chart(data["target"].value_counts())
    st.line_chart(data["age"].value_counts().sort_index())
    st.bar_chart(data.groupby("target")["chol"].mean())

def about_page():
    st.title("ℹ️ About the Project")
    st.write("""
This project predicts heart disease using a Machine Learning model 
(Random Forest). Built with Streamlit for final-year submission.
""")


# -------------------- RENDER SELECTED PAGE --------------------

if st.session_state.page == "Home": home_page()
elif st.session_state.page == "Prediction": prediction_page()
elif st.session_state.page == "Dashboard": dashboard_page()
elif st.session_state.page == "About": about_page()
