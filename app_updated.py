
import io
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


# =========================================================
# 1. PAGE CONFIGURATION AND DESIGN
# =========================================================

st.set_page_config(
    page_title="HealthLens AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Hide Streamlit sidebar and its collapse control */
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}

    .stApp {
        background: #000000;
        color: #eeeeee;
    }

    h1, h2, h3, h4, p, label {
        color: #eeeeee;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: #000000;
        border: 1px solid #292929;
        border-radius: 26px;
        padding: 1.7rem 1.8rem;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: #f3f3f3 !important;
        font-size: 2.4rem;
        margin: 0;
    }

    .hero p {
        color: #c9c9c9 !important;
        font-size: 1.08rem;
        margin: .55rem 0 0;
    }

    .eyebrow {
        color: #0da99b;
        font-weight: 700;
        letter-spacing: .08em;
        font-size: .78rem;
    }

    .feature-card {
        background: #000000;
        border: 1px solid #292929;
        border-radius: 22px;
        padding: 1.4rem;
        min-height: 210px;
    }

    .feature-card h3 {
        color: #f0f0f0;
        margin: .7rem 0;
        font-size: 1.35rem;
    }

    .feature-card p {
        color: #c8c8c8;
        line-height: 1.6;
    }

    .teal-icon {
        color: #0da99b;
        font-size: 2.2rem;
    }

    .stButton button,
    .stDownloadButton button {
        border-radius: 12px;
        font-weight: 650;
        min-height: 44px;
    }

    .stButton button[kind="primary"],
    .stDownloadButton button {
        background: #0b8f84;
        color: white;
        border: 1px solid #0b8f84;
    }

    input, textarea, [data-baseweb="select"] > div {
        background: #111111 !important;
        color: #eeeeee !important;
    }

    [data-testid="stFileUploader"] {
        background: #080808;
        border: 1px dashed #0da99b;
        border-radius: 16px;
        padding: 1rem;
    }

    div[data-testid="stMetric"] {
        background: #090909;
        border: 1px solid #292929;
        padding: .8rem;
        border-radius: 14px;
    }

    hr {
        border-color: #292929;
    }
    
    /* Hide the top toolbar */
    [data-testid="stToolbar"] {
        visibility: hidden;
        height: 0;
    }
    /* Make the header transparent */
    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Add space above page content */
    .block-container {
        padding-top: 2rem;
    }


    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 2. LOAD TRAINED MODELS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_saved_models():
    """Load each trained pipeline and its matching label encoder."""
    loaded = {}

    files = {
        "cardio": ("cardiovascular_model.pkl", "cardiovascular_label_encoder.pkl"),
        "obesity": ("obesity_model.pkl", "obesity_label_encoder.pkl"),
        "diabetes": ("diabetes_model.pkl", "diabetes_label_encoder.pkl"),
    }

    for name, (model_filename, encoder_filename) in files.items():
        model_path = BASE_DIR / model_filename
        encoder_path = BASE_DIR / encoder_filename

        try:
            bundle = joblib.load(model_path)
            # Support either a plain estimator/pipeline or a saved bundle.
            if isinstance(bundle, dict) and "model" in bundle:
                loaded[name] = bundle["model"]
                if "label_encoder" in bundle:
                    loaded[f"{name}_encoder"] = bundle["label_encoder"]
            else:
                loaded[name] = bundle

            if f"{name}_encoder" not in loaded and encoder_path.exists():
                loaded[f"{name}_encoder"] = joblib.load(encoder_path)

        except Exception as exc:
            loaded[f"{name}_error"] = (
                f"Could not load {model_filename}: {exc}"
            )

        # An encoder may be saved separately even if model loading failed.
        if f"{name}_encoder" not in loaded and encoder_path.exists():
            try:
                loaded[f"{name}_encoder"] = joblib.load(encoder_path)
            except Exception as exc:
                loaded[f"{name}_encoder_error"] = str(exc)

    return loaded


models = load_saved_models()


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def decode_prediction(prediction, encoder=None):
    """Convert encoded class IDs back to the original class label when possible."""
    if encoder is None:
        return str(prediction)

    try:
        # LabelEncoder expects integer class IDs; do not re-encode text labels.
        if isinstance(prediction, (int, float)) or hasattr(prediction, "item"):
            class_id = int(prediction)
            if 0 <= class_id < len(encoder.classes_):
                return str(encoder.inverse_transform([class_id])[0])
    except Exception:
        pass

    return str(prediction)


def safe_predict(model, row, encoder=None):
    """Predict from a one-row DataFrame, matching the model's training columns."""
    frame = pd.DataFrame([row])

    expected_columns = getattr(model, "feature_names_in_", None)
    if expected_columns is not None:
        expected_columns = list(expected_columns)
        missing_columns = [column for column in expected_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                "The model expects input columns that are not provided: "
                + ", ".join(missing_columns)
            )
        frame = frame[expected_columns]

    prediction = model.predict(frame)[0]
    decoded_prediction = decode_prediction(prediction, encoder)
    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        probability = float(max(probabilities))

    return decoded_prediction, probability


def is_positive_prediction(prediction):
    """Interpret common binary positive labels without assuming a numeric output."""
    label = str(prediction).strip().lower()
    return label in {"1", "yes", "true", "positive", "diabetes", "cardio", "disease", "higher risk"}


def estimate_blood_pressure(systolic, diastolic):
    """Simplified adult blood-pressure screening thresholds."""

    if systolic >= 180 or diastolic >= 120:
        return (
            "Very high reading",
            "Seek urgent medical advice. If you have chest pain, "
            "shortness of breath, weakness, confusion, or vision "
            "changes, seek emergency care.",
        )

    if systolic >= 140 or diastolic >= 90:
        return (
            "High reading (stage 2 range)",
            "Arrange professional assessment and repeat measurements.",
        )

    if systolic >= 130 or diastolic >= 80:
        return (
            "High reading (stage 1 range)",
            "Repeat measurements and discuss the results with "
            "a healthcare professional.",
        )

    if systolic >= 120 and diastolic < 80:
        return (
            "Elevated systolic reading",
            "Recheck under proper measurement conditions.",
        )

    return (
        "Not elevated by these thresholds",
        "One reading cannot rule out hypertension. "
        "Use repeated measurements and clinical advice.",
    )


def assess_total_cholesterol(total_cholesterol):
    """Educational total-cholesterol categories in mg/dL."""

    if total_cholesterol < 200:
        return (
            "Desirable range",
            "Below 200 mg/dL. This alone does not establish overall "
            "lipid or cardiovascular health.",
        )

    if total_cholesterol < 240:
        return (
            "Borderline-high range",
            "Discuss the result with a healthcare professional "
            "who can interpret the complete lipid panel.",
        )

    return (
        "High range",
        "Discuss this result with a healthcare professional. "
        "Interpretation also depends on LDL, HDL, triglycerides, "
        "and overall risk.",
    )


def make_pdf(report):
    """Create a downloadable PDF summary."""

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42,
    )

    styles = getSampleStyleSheet()

    table_data = [["Item", "Value"]]
    for key, value in report.items():
        table_data.append([str(key), str(value)])

    report_table = Table(
        table_data,
        colWidths=[180, 300],
        repeatRows=1,
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123448")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6dee8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [colors.white, colors.HexColor("#f3f7fb")],
            ),
            ("PADDING", (0, 0), (-1, -1), 7),
        ]),
    )

    story = [
        Paragraph("HealthLens AI — Screening Summary", styles["Title"]),
        Spacer(1, 8),
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            styles["Normal"],
        ),
        Spacer(1, 14),
        Paragraph(
            "This educational prototype is not a medical diagnosis "
            "or a clinically validated decision tool. Do not use it "
            "to make treatment decisions. Discuss concerns with a "
            "qualified healthcare professional.",
            styles["BodyText"],
        ),
        Spacer(1, 16),
        Paragraph("Assessment details", styles["Heading2"]),
        Spacer(1, 8),
        report_table,
        Spacer(1, 16),
        Paragraph("Limitations", styles["Heading2"]),
        Paragraph(
            "The machine-learning models were trained on specific "
            "datasets and have not been clinically validated. "
            "Blood-pressure and cholesterol outputs use simplified "
            "thresholds. Model probabilities are not the same as "
            "clinical disease probabilities.",
            styles["BodyText"],
        ),
    ]

    document.build(story)
    return buffer.getvalue()


# Keep results and report between Streamlit reruns.
if "last_results" not in st.session_state:
    st.session_state["last_results"] = {}

if "report" not in st.session_state:
    st.session_state["report"] = {}

if "last_notes" not in st.session_state:
    st.session_state["last_notes"] = {}

# =========================================================
# 4. HOME PAGE — ONLY THE MAIN TWO CARDS
# =========================================================

if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Overview"

page = st.session_state["selected_page"]

if page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <div class="teal-icon">♡</div>
            <h1>HealthLens AI</h1>
            <p>Your health data, intelligently analyzed.</p>
            <p>
                Understand health indicators, explore potential risks,
                and discover the factors influencing your results.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            """
            <div class="feature-card">
                <div class="teal-icon">▤</div>
                <h3>Enter Health Data</h3>
                <p>
                    Manually enter health measurements and lifestyle
                    information.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Manual assessment",
            key="go_manual",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["selected_page"] = "Manual health assessment"
            st.rerun()

    with right:
        st.markdown(
            """
            <div class="feature-card">
                <div class="teal-icon">⇧</div>
                <h3>Upload Health Reports</h3>
                <p>
                    Upload blood reports or other supported
                    health documents.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Report analysis",
            key="go_upload",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["selected_page"] = "Upload health reports"
            st.rerun()

    st.caption("Concept prototype • Educational use only")


# =========================================================
# 5. MANUAL HEALTH ASSESSMENT
# =========================================================

elif page == "Manual health assessment":

    if st.button("← Back to Home"):
        st.session_state["selected_page"] = "Overview"
        st.rerun()

    st.title("Enter Health Data")
    st.write("Enter your measurements and lifestyle details.")

    with st.form("health_assessment_form"):

        st.markdown("### A. Basic measurements")

        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input(
                "Age (years)", min_value=18, max_value=100,
                value=None, placeholder="Enter age"
            )
            gender_label = st.selectbox(
                "Gender", ["Select...", "Female", "Male"]
            )
            height_m = st.number_input(
                "Height (metres)", min_value=1.0, max_value=2.5,
                value=None, placeholder="e.g. 1.70", step=0.01
            )

        with c2:
            weight = st.number_input(
                "Weight (kg)", min_value=20.0, max_value=300.0,
                value=None, placeholder="Enter weight", step=0.5
            )
            systolic = st.number_input(
                "Systolic blood pressure (mmHg)",
                min_value=60, max_value=250,
                value=None, placeholder="e.g. 120"
            )
            diastolic = st.number_input(
                "Diastolic blood pressure (mmHg)",
                min_value=30, max_value=160,
                value=None, placeholder="e.g. 80"
            )

        with c3:
            total_cholesterol = st.number_input(
                "Total cholesterol (mg/dL)",
                min_value=80, max_value=500,
                value=None, placeholder="Enter lab result"
            )
            cardio_cholesterol_code = st.selectbox(
                "Cardiovascular dataset cholesterol category",
                ["Select...", 1, 2, 3]
            )
            glucose_category = st.selectbox(
                "Cardiovascular dataset glucose category",
                ["Select...", 1, 2, 3]
            )

        bmi = (
            weight / (height_m * height_m)
            if height_m is not None and weight is not None
            else None
        )

        if bmi is not None:
            st.caption(f"Calculated BMI: {bmi:.1f} kg/m²")

        st.markdown("### B. Diabetes model inputs")

        st.caption(
            "Use measured values where possible. Do not guess "
            "laboratory results. This model was trained on a "
            "specific population and may not generalize to everyone."
        )

        d1, d2, d3 = st.columns(3)

        with d1:
            pregnancies = st.number_input(
                "Pregnancies", min_value=0, max_value=20,
                value=None, placeholder="Enter number"
            )
            glucose = st.number_input(
                "Glucose (mg/dL)", min_value=1, max_value=350,
                value=None, placeholder="Enter lab result"
            )
            blood_pressure_diabetes = st.number_input(
                "Diabetes dataset BP (mmHg)",
                min_value=1, max_value=200,
                value=None, placeholder="Enter measurement"
            )

        with d2:
            skin_thickness = st.number_input(
                "Skin thickness (mm)", min_value=1, max_value=100,
                value=None, placeholder="Enter measurement"
            )
            insulin = st.number_input(
                "Insulin (mu U/mL)", min_value=1, max_value=900,
                value=None, placeholder="Enter lab result"
            )
            diabetes_bmi = st.number_input(
                "Diabetes dataset BMI", min_value=1.0, max_value=80.0,
                value=None, placeholder="Enter BMI", step=0.1
            )

        with d3:
            diabetes_pedigree = st.number_input(
                "Diabetes pedigree function",
                min_value=0.0, max_value=3.0,
                value=None, placeholder="Enter value", step=0.05
            )

        st.markdown("### C. Lifestyle information")

        o1, o2, o3 = st.columns(3)

        with o1:
            family_history = st.selectbox(
                "Family history of overweight", ["Select...", "yes", "no"]
            )
            frequent_calories = st.selectbox(
                "Frequent high-calorie food", ["Select...", "yes", "no"]
            )
            vegetables = st.selectbox(
                "Vegetable intake (FCVC)", ["Select...", 1.0, 2.0, 3.0]
            )
            meals = st.selectbox(
                "Main meals per day (NCP)",
                ["Select...", 1.0, 2.0, 3.0, 4.0]
            )
            between_meals = st.selectbox(
                "Food between meals (CAEC)",
                ["Select...", "no", "Sometimes", "Frequently", "Always"]
            )

        with o2:
            smoke = st.selectbox(
                "Smoking", ["Select...", "no", "yes"]
            )
            water = st.selectbox(
                "Daily water intake (CH2O)", ["Select...", 1.0, 2.0, 3.0]
            )
            calorie_monitoring = st.selectbox(
                "Calorie monitoring (SCC)", ["Select...", "no", "yes"]
            )
            physical_activity = st.selectbox(
                "Physical activity (FAF)", ["Select...", 0.0, 1.0, 2.0, 3.0]
            )
            screen_time = st.selectbox(
                "Technology use (TUE)", ["Select...", 0.0, 1.0, 2.0]
            )

        with o3:
            alcohol = st.selectbox(
                "Alcohol consumption (CALC)",
                ["Select...", "no", "Sometimes", "Frequently", "Always"]
            )
            transport = st.selectbox(
                "Usual transport",
                [
                    "Select...", "Public_Transportation", "Automobile",
                    "Walking", "Motorbike", "Bike"
                ]
            )
            smoking_current = st.selectbox(
                "Smoking (cardiovascular model)", ["Select...", "No", "Yes"]
            )
            alcohol_cardio = st.selectbox(
                "Alcohol use (cardiovascular model)", ["Select...", "No", "Yes"]
            )
            active_cardio = st.selectbox(
                "Physically active", ["Select...", "Yes", "No"]
            )

        submitted = st.form_submit_button(
            "Analyze health indicators",
            type="primary",
            use_container_width=True
        )

    if submitted:

        required_values = {
            "Age": age,
            "Gender": gender_label,
            "Height": height_m,
            "Weight": weight,
            "Systolic blood pressure": systolic,
            "Diastolic blood pressure": diastolic,
            "Total cholesterol": total_cholesterol,
            "Cholesterol category": cardio_cholesterol_code,
            "Glucose category": glucose_category,
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "Diabetes blood pressure": blood_pressure_diabetes,
            "Skin thickness": skin_thickness,
            "Insulin": insulin,
            "Diabetes BMI": diabetes_bmi,
            "Diabetes pedigree function": diabetes_pedigree,
            "Family history": family_history,
            "High-calorie food": frequent_calories,
            "Vegetable intake": vegetables,
            "Main meals": meals,
            "Between-meal food": between_meals,
            "Smoking": smoke,
            "Water intake": water,
            "Calorie monitoring": calorie_monitoring,
            "Physical activity": physical_activity,
            "Technology use": screen_time,
            "Alcohol consumption": alcohol,
            "Transport": transport,
            "Cardiovascular smoking": smoking_current,
            "Cardiovascular alcohol": alcohol_cardio,
            "Cardiovascular activity": active_cardio,
        }

        missing = [
            name for name, value in required_values.items()
            if value is None or value == "Select..."
        ]

        if missing:
            st.error(
                "Please complete all fields before analysis: "
                + ", ".join(missing)
            )

        elif systolic <= diastolic:
            st.error(
                "Systolic blood pressure must be greater than "
                "diastolic blood pressure."
            )

        else:
            # Use the existing model and assessment logic below.
            # Values are validated before they are passed to the models.

            report = {
                "Age (years)": age,
                "Gender": gender_label,
                "Height (m)": height_m,
                "Weight (kg)": weight,
                "Calculated BMI": round(bmi, 1),
                "Blood pressure (mmHg)": f"{systolic}/{diastolic}",
                "Total cholesterol (mg/dL)": total_cholesterol,
            }

            results = {}

            if "diabetes" in models:
                diabetes_row = {
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": blood_pressure_diabetes,
                    "SkinThickness": skin_thickness,
                    "Insulin": insulin,
                    "BMI": diabetes_bmi,
                    "DiabetesPedigreeFunction": diabetes_pedigree,
                    "Age": age,
                }

                try:
                    pred, probability = safe_predict(
                        models["diabetes"], diabetes_row, models.get("diabetes_encoder")
                    )
                    results["Diabetes"] = (
                        f"Positive model class ({pred})"
                        if is_positive_prediction(pred)
                        else f"Negative model class ({pred})"
                    )
                    report["Diabetes model output"] = results["Diabetes"]
                    report["Diabetes top-class probability"] = (
                        f"{probability:.1%}"
                        if probability is not None else "Not available"
                    )
                except Exception as exc:
                    results["Diabetes"] = f"Could not calculate: {exc}"

            if "cardio" in models:
                cardio_row = {
                    "age": age,
                    "gender": 1 if gender_label == "Female" else 2,
                    "height": int(round(height_m * 100)),
                    "weight": weight,
                    "ap_hi": systolic,
                    "ap_lo": diastolic,
                    "cholesterol": int(cardio_cholesterol_code),
                    "gluc": int(glucose_category),
                    "smoke": int(smoking_current == "Yes"),
                    "alco": int(alcohol_cardio == "Yes"),
                    "active": int(active_cardio == "Yes"),
                }

                try:
                    pred, probability = safe_predict(
                        models["cardio"], cardio_row, models.get("cardio_encoder")
                    )
                    results["Cardiovascular"] = (
                        f"Positive model class ({pred})"
                        if is_positive_prediction(pred)
                        else f"Negative model class ({pred})"
                    )
                    report["Cardiovascular model output"] = (
                        results["Cardiovascular"]
                    )
                    report["Cardiovascular top-class probability"] = (
                        f"{probability:.1%}"
                        if probability is not None else "Not available"
                    )
                except Exception as exc:
                    results["Cardiovascular"] = f"Could not calculate: {exc}"

            if "obesity" in models:
                obesity_row = {
                    "Gender": gender_label,
                    "Age": age,
                    "Height": height_m,
                    "Weight": weight,
                    "family_history": family_history,
                    "FAVC": frequent_calories,
                    "FCVC": float(vegetables),
                    "NCP": float(meals),
                    "CAEC": between_meals,
                    "SMOKE": smoke,
                    "CH2O": float(water),
                    "SCC": calorie_monitoring,
                    "FAF": float(physical_activity),
                    "TUE": float(screen_time),
                    "CALC": alcohol,
                    "MTRANS": transport,
                }

                try:
                    pred, probability = safe_predict(
                        models["obesity"], obesity_row, models.get("obesity_encoder")
                    )
                    results["Obesity"] = str(pred).replace("_", " ")
                    report["Obesity model output"] = results["Obesity"]
                    report["Obesity top-class probability"] = (
                        f"{probability:.1%}"
                        if probability is not None else "Not available"
                    )
                except Exception as exc:
                    results["Obesity"] = f"Could not calculate: {exc}"

            bp_label, bp_note = estimate_blood_pressure(systolic, diastolic)
            results["Blood pressure assessment"] = bp_label
            report["Blood pressure assessment"] = bp_label
            report["Blood pressure note"] = bp_note

            chol_label, chol_note = assess_total_cholesterol(
                total_cholesterol
            )
            results["Cholesterol assessment"] = chol_label
            report["Cholesterol assessment"] = chol_label
            report["Cholesterol note"] = chol_note

            st.session_state["report"] = report
            st.session_state["last_results"] = results
            st.session_state["last_notes"] = {
                "Blood pressure": bp_note,
                "Cholesterol": chol_note,
            }

            st.success("Analysis completed.")

            for model_name, friendly_name in [
                ("diabetes", "Diabetes"),
                ("cardio", "Cardiovascular"),
                ("obesity", "Obesity"),
            ]:
                if f"{model_name}_error" in models:
                    st.warning(
                        f"{friendly_name} model unavailable: "
                        f"{models[f'{model_name}_error']}"
                    )
                elif model_name not in models:
                    st.warning(
                        f"{friendly_name} model file was not loaded. "
                        "Check that its .pkl file is in the same folder as app.py."
                    )

            st.markdown("---")
            st.markdown("## Your screening summary")

            for name, value in results.items():
                st.markdown(f"**{name}:** {value}")

            st.caption(
                "Educational prototype only. These outputs are not "
                "diagnoses and should not replace professional care."
            )

            pdf_data = make_pdf(report)

            st.download_button(
                "Download screening summary (PDF)",
                data=pdf_data,
                file_name="healthlens_ai_screening_summary.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )



# =========================================================
# 7. UPLOAD HEALTH REPORTS
# =========================================================

elif page == "Upload health reports":

    if st.button("← Back to Home"):
        st.session_state["selected_page"] = "Overview"
        st.rerun()

    st.title("Upload Health Reports")
    st.write(
        "Select a blood report or another health document. "
        "Blood reports are the first priority."
    )

    report_type = st.selectbox(
        "Document type",
        [
            "Blood report",
            "Prescription",
            "Other health document",
        ],
    )

    uploaded_file = st.file_uploader(
        "＋ Add a health report",
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_file:
        st.success(f"Selected: {uploaded_file.name}")

        st.caption(
            f"Document type: {report_type} • "
            f"File size: {uploaded_file.size / 1024:.1f} KB"
        )

        st.warning(
            "Automatic report reading and analysis are not "
            "implemented yet. Selecting a file does not mean "
            "its contents have been analyzed."
        )

        if st.button(
            "Continue to manual assessment",
            type="primary",
        ):
            st.session_state["selected_page"] = (
                "Manual health assessment"
            )
            st.rerun()

    else:
        st.info(
            "Upload a PDF, JPG, or PNG file to begin."
        )


# =========================================================
# 8. ABOUT AND LIMITATIONS
# =========================================================

elif page == "About & limitations":

    if st.button("← Back to Home"):
        st.session_state["selected_page"] = "Overview"
        st.rerun()

    st.title("About HealthLens AI")

    st.write(
        "HealthLens AI is an educational prototype demonstrating "
        "machine-learning predictions and measurement-based "
        "health assessments."
    )

    st.markdown("### Current indicators")

    st.markdown(
        """
        - **Obesity:** trained machine-learning model.
        - **Diabetes:** trained machine-learning model.
        - **Cardiovascular disease:** trained machine-learning model.
        - **Blood pressure:** simplified measurement-based assessment.
        - **Cholesterol:** simplified total-cholesterol assessment.
        """
    )

    st.markdown("### Important limitations")

    st.markdown(
        """
        - These models have not been clinically validated.
        - The diabetes dataset represents a specific population
          and may not generalize to everyone.
        - Model probabilities are not necessarily calibrated
          clinical risk estimates.
        - Blood pressure should be interpreted using repeated
          measurements and appropriate clinical guidance.
        - Total cholesterol alone does not provide a complete
          picture of lipid or cardiovascular health.
        - Uploaded reports are not automatically analyzed yet.
        - Do not use these outputs to make treatment decisions.
        """
    )

    st.markdown("### Required model files")

    st.code(
        "app.py\n"
        "diabetes_model.pkl\n"
        "obesity_model.pkl\n"
        "cardiovascular_model.pkl"
    )
