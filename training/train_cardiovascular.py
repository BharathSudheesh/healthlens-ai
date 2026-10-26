
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Load cardiovascular dataset
df = pd.read_csv("cardio_train.csv", sep=None, engine="python")
df.columns = df.columns.str.strip()

target = next((c for c in ["cardio", "Cardio", "target", "Target"]
               if c in df.columns), None)
if target is None:
    raise ValueError(f"Target column not found. Columns: {df.columns.tolist()}")

df = df.dropna(subset=[target])
X = df.drop(columns=[target])
y = df[target]

# Convert age from days to years, if applicable
if "age" in X.columns.str.lower():
    age_col = next(c for c in X.columns if c.lower() == "age")
    if X[age_col].median() > 150:
        X[age_col] = X[age_col] / 365.25

# Remove identifier columns
X = X.drop(columns=[c for c in X.columns
                    if c.lower() in ["id", "patient_id"]], errors="ignore")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y.astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

numeric = X.select_dtypes(include=np.number).columns.tolist()
categorical = X.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        objective="binary:logistic" if len(encoder.classes_) == 2 else "multi:softprob",
        eval_metric="logloss" if len(encoder.classes_) == 2 else "mlogloss",
        random_state=42, n_jobs=-1
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    results.append({
        "Algorithm": name,
        "CV Macro-F1": scores.mean(),
        "Test Accuracy": accuracy_score(y_test, predictions),
        "Test Macro-F1": f1_score(y_test, predictions, average="macro")
    })

    print(f"\n{name}")
    print(f"CV Macro-F1: {scores.mean():.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(
        y_test, predictions, labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_, zero_division=0
    ))

results_df = pd.DataFrame(results).sort_values(
    "CV Macro-F1", ascending=False
)
print("\nFINAL ALGORITHM COMPARISON")
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["Algorithm"]
best_model = models[best_name]

best_pipeline = Pipeline([("prep", preprocessor), ("model", best_model)])
best_pipeline.fit(X, y_encoded)

joblib.dump(best_pipeline, "cardiovascular_model.pkl")
joblib.dump(encoder, "cardiovascular_label_encoder.pkl")

print(f"\nBest algorithm: {best_name}")
print("Saved cardiovascular_model.pkl and cardiovascular_label_encoder.pkl")
