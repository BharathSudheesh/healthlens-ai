
# ============================================
# HealthLens AI - Obesity Model Training
# Compare 6 algorithms and save the best model
# ============================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# 1. Load dataset
df = pd.read_csv("Obesity prediction.csv")
df.columns = df.columns.str.strip()

target_column = "Obesity"

if target_column not in df.columns:
    raise ValueError(
        f"Target '{target_column}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

df = df.dropna(subset=[target_column])

X = df.drop(columns=[target_column])
y = df[target_column].astype(str)


# 2. Encode target labels for XGBoost
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# 4. Identify numerical and categorical columns
numeric_columns = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=np.number
).columns.tolist()


# 5. Preprocess the features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_columns
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_columns
        )
    ]
)


# 6. Define six algorithms
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "SVM": SVC(),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
}


# 7. Compare algorithms using cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

print("\nOBESITY MODEL COMPARISON")
print("=" * 60)

for name, model in models.items():

    print(f"\nTraining: {name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1
    )

    # Fit on training data
    pipeline.fit(X_train, y_train)

    # Evaluate on unseen test data
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    results.append({
        "Algorithm": name,
        "CV Macro-F1": scores.mean(),
        "CV Std": scores.std(),
        "Test Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1
    })

    print(f"CV Macro-F1: {scores.mean():.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro-F1: {macro_f1:.4f}")


# 8. Display final comparison
results_df = pd.DataFrame(results).sort_values(
    by="CV Macro-F1",
    ascending=False
)

print("\n" + "=" * 90)
print("FINAL ALGORITHM COMPARISON")
print("=" * 90)
print(results_df.to_string(index=False))


# 9. Select the winner by cross-validation Macro-F1
best_name = results_df.iloc[0]["Algorithm"]

print(f"\nBest algorithm: {best_name}")


# 10. Train the winning model on the complete dataset
best_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", models[best_name])
])

best_pipeline.fit(X, y_encoded)


# 11. Save model and label encoder
joblib.dump(best_pipeline, "obesity_model.pkl")
joblib.dump(label_encoder, "obesity_label_encoder.pkl")

print("\nSaved successfully:")
print("1. obesity_model.pkl")
print("2. obesity_label_encoder.pkl")
print(f"\nWinning algorithm: {best_name}")
print("Training and comparison completed!")
