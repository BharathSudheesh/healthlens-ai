
"""
SHAP explanation helper for HealthLens AI.
Displays simple, readable descriptions instead of raw SHAP values.
"""

import numpy as np
import pandas as pd


FEATURE_NAMES = {
    "age": "Age",
    "gender": "Gender",
    "height": "Height",
    "weight": "Weight",
    "ap_hi": "Systolic blood pressure",
    "ap_lo": "Diastolic blood pressure",
    "cholesterol": "Cholesterol",
    "gluc": "Glucose",
    "smoke": "Smoking",
    "alco": "Alcohol consumption",
    "active": "Physical activity",
    "Pregnancies": "Pregnancy history",
    "Glucose": "Glucose level",
    "BloodPressure": "Blood pressure",
    "SkinThickness": "Skin-fold thickness",
    "Insulin": "Insulin level",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Family-history indicator",
    "Age": "Age",
    "Gender": "Gender",
    "Height": "Height",
    "Weight": "Weight",
    "family_history": "Family history of overweight",
    "FAVC": "Frequent high-calorie food",
    "FCVC": "Vegetable intake",
    "NCP": "Main meals per day",
    "CAEC": "Food between meals",
    "SMOKE": "Smoking",
    "CH2O": "Water intake",
    "SCC": "Calorie monitoring",
    "FAF": "Physical activity",
    "TUE": "Screen time",
    "CALC": "Alcohol consumption",
    "MTRANS": "Usual transport",
}


def get_feature_name(name):
    """Remove preprocessing prefixes and return a readable feature name."""
    name = str(name)
    clean_name = name.split("__")[-1]

    return FEATURE_NAMES.get(
        name,
        FEATURE_NAMES.get(clean_name, clean_name.replace("_", " "))
    )


def describe_feature(name, value):
    """Convert an input value into a simple description."""
    feature = str(name).split("__")[-1].lower()

    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value).strip().lower()

        if feature in {"favc", "family_history", "smoke", "smoking"}:
            if text in {"yes", "true"}:
                if feature == "favc":
                    return "Frequent high-calorie food consumption"
                if feature == "family_history":
                    return "Family history of overweight"
                return "Smoking reported"

        if text in {"no", "false"}:
            if feature == "favc":
                return "No frequent high-calorie food reported"
            if feature == "family_history":
                return "No family history of overweight reported"
            if feature in {"smoke", "smoking"}:
                return "No smoking reported"

        if text:
            return f"{get_feature_name(name)}: {text.replace('_', ' ')}"

        return get_feature_name(name)

    if feature == "glucose":
        if number >= 126:
            return "High glucose level"
        if number >= 100:
            return "Glucose level in the 100–125 mg/dL range"
        return "Glucose level below 100 mg/dL"

    if feature in {"bmi", "numeric_bmi"}:
        if number >= 30:
            return "High BMI"
        if number >= 25:
            return "BMI in the overweight range"
        return "BMI below 25"

    if feature in {"ap_hi", "systolic"}:
        if number >= 140:
            return "High systolic blood pressure"
        if number >= 130:
            return "Systolic blood pressure in the 130–139 range"
        if number >= 120:
            return "Elevated systolic blood pressure"
        return "Systolic blood pressure below 120 mmHg"

    if feature in {"ap_lo", "diastolic"}:
        if number >= 90:
            return "High diastolic blood pressure"
        if number >= 80:
            return "Diastolic blood pressure in the 80–89 range"
        return "Diastolic blood pressure below 80 mmHg"

    if feature == "cholesterol":
        if number >= 3:
            return "High cholesterol category"
        if number >= 2:
            return "Elevated cholesterol category"
        return "Cholesterol category recorded as normal"

    if feature == "ch2o":
        if number <= 1:
            return "Low reported water intake"
        if number >= 3:
            return "Higher reported water intake"
        return "Moderate reported water intake"

    if feature == "faf":
        if number <= 0:
            return "No reported physical activity"
        if number < 2:
            return "Low reported physical activity"
        return "Higher reported physical activity"

    if feature == "favc":
        if number == 1:
            return "Frequent high-calorie food consumption"

    if feature in {"weight", "numeric_weight"}:
        return f"Weight: {number:g} kg"

    if feature in {"height", "numeric_height"}:
        return f"Height: {number:g} m"

    if feature == "age":
        return f"Age: {number:g} years"

    return f"{get_feature_name(name)}: {number:g}"


def explain_prediction(model, row: dict, max_features: int = 5):
    """
    Return a dictionary containing readable explanations based on SHAP.
    The model's prediction itself is not changed.
    """

    try:
        import shap
    except ImportError:
        return {
            "available": False,
            "summary": "SHAP is unavailable.",
            "details": [],
        }

    try:
        frame = pd.DataFrame([row])

        expected_columns = getattr(model, "feature_names_in_", None)
        if expected_columns is not None:
            expected_columns = list(expected_columns)

            missing = [
                col for col in expected_columns
                if col not in frame.columns
            ]

            if missing:
                return {
                    "available": False,
                    "summary": "A reliable explanation is unavailable.",
                    "details": [],
                }

            frame = frame[expected_columns]

        estimator = model
        model_input = frame
        feature_names = list(frame.columns)

        # Handle sklearn pipelines.
        if hasattr(model, "steps") and model.steps:
            estimator = model.steps[-1][1]

            if len(model.steps) > 1:
                transformer = model[:-1]
                model_input = transformer.transform(frame)

                try:
                    feature_names = list(
                        transformer.get_feature_names_out()
                    )
                except Exception:
                    feature_names = [
                        f"feature_{i + 1}"
                        for i in range(model_input.shape[1])
                    ]

        class_name = type(estimator).__name__.lower()
        module_name = type(estimator).__module__.lower()

        is_tree_model = (
            "tree" in module_name
            or any(word in class_name for word in (
                "forest",
                "tree",
                "xgb",
                "lgbm",
                "catboost",
                "gradientboosting",
                "extratrees",
                "randomforest",
            ))
        )

        # Tree models can be explained directly.
        if not is_tree_model:
            return {
                "available": False,
                "summary": "A reliable explanation is unavailable for this model.",
                "details": [],
            }

        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(model_input)

        # Normalize common SHAP output formats.
        if isinstance(shap_values, list):
            class_index = 0

            try:
                prediction = estimator.predict(model_input)[0]
                classes = list(estimator.classes_)

                if prediction in classes:
                    class_index = classes.index(prediction)
            except Exception:
                pass

            values = np.asarray(
                shap_values[min(class_index, len(shap_values) - 1)]
            )[0]

        else:
            values = np.asarray(shap_values)

            if values.ndim == 3:
                values = values[0]

                if values.ndim == 2:
                    class_index = 0

                    try:
                        prediction = estimator.predict(model_input)[0]
                        classes = list(estimator.classes_)

                        if prediction in classes:
                            class_index = classes.index(prediction)
                    except Exception:
                        pass

                    values = values[:, min(
                        class_index, values.shape[-1] - 1
                    )]

            elif values.ndim == 2:
                values = values[0]

        values = np.asarray(values).reshape(-1)

        if len(values) != len(feature_names):
            return {
                "available": False,
                "summary": "A reliable explanation is unavailable.",
                "details": [],
            }

        # Rank the features by the absolute size of their SHAP values.
        ranked_indices = np.argsort(np.abs(values))[::-1]

        details = []
        seen = set()

        for index in ranked_indices:
            if len(details) >= max_features:
                break

            contribution = float(values[index])

            # Skip features with effectively no contribution.
            if abs(contribution) < 1e-8:
                continue

            feature = feature_names[index]
            clean_name = str(feature).split("__")[-1]

            value = row.get(clean_name, row.get(feature))

            if value is None:
                continue

            reason = describe_feature(feature, value)

            if reason not in seen:
                details.append(reason)
                seen.add(reason)

        if not details:
            return {
                "available": False,
                "summary": "No clear explanation is available.",
                "details": [],
            }

        return {
            "available": True,
            "summary": "Possible factors associated with the model's prediction:",
            "details": details,
        }

    except Exception:
        return {
            "available": False,
            "summary": "A reliable explanation could not be generated.",
            "details": [],
        }
