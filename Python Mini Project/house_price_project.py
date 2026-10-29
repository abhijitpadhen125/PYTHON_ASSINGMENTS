
# ============================================================
# MINI PROJECT: HOUSE PRICE ANALYSIS AND PREDICTION
# Subject: Python Programming for Data Science
# CO1: NumPy and Pandas
# CO2: Data Preprocessing
# CO3: Data Visualization
# CO4: Regression and Classification
# CO5: Model Evaluation
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# -------------------- SETTINGS --------------------

DATA_FILE = os.path.join("data", "Pune_House_Data.csv")
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

print("=" * 55)
print("HOUSE PRICE ANALYSIS AND PREDICTION")
print("=" * 55)

# -------------------- CO1: NUMPY --------------------

print("\nCO1: NUMPY OPERATIONS")

numbers = np.array([10, 20, 30, 40, 50])

print("Array:", numbers)
print("First element:", numbers[0])
print("Slicing:", numbers[:3])
print("Reshape:\n", numbers.reshape(5, 1))
print("Vectorized addition:", numbers + 5)
print("Mean:", np.mean(numbers))
print("Median:", np.median(numbers))
print("Standard deviation:", np.std(numbers))

# -------------------- CO1: PANDAS --------------------

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"\nDataset not found: {DATA_FILE}\n"
        "Create a data folder and place Pune_House_Data.csv inside it."
    )

df = pd.read_csv(DATA_FILE)

print("\nFirst five records:\n", df.head())
print("\nDataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nDataset information:")
df.info()
print("\nStatistical summary:\n", df.describe(include="all"))

# -------------------- CO2: PREPROCESSING --------------------

print("\nCO2: DATA PREPROCESSING")

original_rows = len(df)
df = df.drop_duplicates().copy()

print("Duplicate rows removed:", original_rows - len(df))

# Convert relevant columns to numeric values
for col in ["price", "bath", "balcony"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Extract bedroom count from values such as '2 BHK'
if "size" in df.columns:
    df["bhk"] = pd.to_numeric(
        df["size"].astype("string").str.extract(r"(\d+)")[0],
        errors="coerce"
    )

# Convert total area to numeric; use midpoint for ranges
def convert_area(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().replace(",", "")

    try:
        if "-" in text:
            parts = text.split("-")
            if len(parts) == 2:
                return (float(parts[0]) + float(parts[1])) / 2
        return float(text)
    except (ValueError, TypeError):
        return np.nan


if "total_sqft" in df.columns:
    df["total_sqft"] = df["total_sqft"].apply(convert_area)
    df.loc[df["total_sqft"] <= 0, "total_sqft"] = np.nan

if "bhk" in df.columns:
    df.loc[df["bhk"] <= 0, "bhk"] = np.nan

# Exclude records without a valid target
df = df[df["price"].notna() & (df["price"] > 0)].copy()

print("Cleaned dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Filtering
print("\nProperties with 2 or more bedrooms:")
print(df[df["bhk"] >= 2].head())

# Sorting
print("\nFive highest prices:")
print(df.nlargest(5, "price")[
    ["site_location", "bhk", "total_sqft", "price"]
])

# Grouping and aggregation
location_summary = df.groupby("site_location")["price"].agg(
    ["mean", "median", "count"]
).sort_values("mean", ascending=False)

print("\nLocation-wise summary:")
print(location_summary.head(10))

location_summary.to_csv(
    os.path.join(OUTPUT_DIR, "location_summary.csv")
)
df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), index=False)

# -------------------- CO3: VISUALIZATION --------------------

print("\nCO3: CREATING GRAPHS")

# Graph 1: Price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=30, kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Price (original dataset units)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "price_distribution.png"))
plt.show()
plt.close()

# Graph 2: Bedroom counts
plt.figure(figsize=(8, 5))
df["bhk"].value_counts().sort_index().head(10).plot(kind="bar")
plt.title("Properties by Number of Bedrooms")
plt.xlabel("Bedrooms (BHK)")
plt.ylabel("Number of properties")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "bedroom_counts.png"))
plt.show()
plt.close()

# Graph 3: Area versus price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="total_sqft", y="price", alpha=0.5)
plt.title("Area versus House Price")
plt.xlabel("Area (square feet)")
plt.ylabel("Price (original dataset units)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "area_vs_price.png"))
plt.show()
plt.close()

# Graph 4: Correlation heatmap
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.show()
plt.close()

# -------------------- CO4: MODEL PREPARATION --------------------

print("\nCO4: MACHINE LEARNING")

numeric_candidates = ["total_sqft", "bath", "balcony", "bhk"]
categorical_candidates = [
    "site_location", "area_type", "availability"
]

numeric_features = [
    c for c in numeric_candidates if c in df.columns
]
categorical_features = [
    c for c in categorical_candidates if c in df.columns
]
features = numeric_features + categorical_features

if not numeric_features:
    raise ValueError("No suitable numerical input features found.")

X = df[features].copy()
y = df["price"].copy()

if len(X) < 20:
    raise ValueError("The dataset has too few valid records.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Features:", features)

def create_preprocessor():
    transformers = []

    if numeric_features:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(
            ("numeric", numeric_pipe, numeric_features)
        )

    if categorical_features:
        category_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(
            ("categorical", category_pipe, categorical_features)
        )

    return ColumnTransformer(transformers=transformers)

# -------------------- REGRESSION --------------------

regression_model = Pipeline([
    ("preprocessing", create_preprocessor()),
    ("model", LinearRegression())
])

regression_model.fit(X_train, y_train)
predictions = regression_model.predict(X_test)

print("\nLINEAR REGRESSION RESULTS")
print("MAE:", round(mean_absolute_error(y_test, predictions), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, predictions)), 3))
print("R2 score:", round(r2_score(y_test, predictions), 3))

comparison = pd.DataFrame({
    "Actual Price": y_test.to_numpy(),
    "Predicted Price": predictions
})

print("\nActual and predicted prices:\n", comparison.head(10))
comparison.to_csv(
    os.path.join(OUTPUT_DIR, "prediction_comparison.csv"),
    index=False
)

plt.figure(figsize=(7, 5))
plt.scatter(y_test, predictions, alpha=0.5)
low = min(y_test.min(), predictions.min())
high = max(y_test.max(), predictions.max())
plt.plot([low, high], [low, high], "r--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual versus Predicted Prices")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "regression_results.png"))
plt.show()
plt.close()

# -------------------- CLASSIFICATION --------------------

# Define relative price categories from training prices only
lower = y_train.quantile(1 / 3)
upper = y_train.quantile(2 / 3)

def classify_price(values):
    return np.where(
        values <= lower,
        "Budget",
        np.where(values <= upper, "Mid-Range", "Premium")
    )

y_train_class = classify_price(y_train)
y_test_class = classify_price(y_test)

classification_model = Pipeline([
    ("preprocessing", create_preprocessor()),
    ("model", LogisticRegression(max_iter=2000))
])

classification_model.fit(X_train, y_train_class)
class_predictions = classification_model.predict(X_test)

print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", round(
    accuracy_score(y_test_class, class_predictions), 3
))
print("\nClassification report:")
print(classification_report(
    y_test_class, class_predictions, zero_division=0
))

ConfusionMatrixDisplay.from_predictions(
    y_test_class,
    class_predictions,
    labels=["Budget", "Mid-Range", "Premium"],
    cmap="Blues"
)
plt.title("Classification Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()
plt.close()

# -------------------- CO5: FINAL PREDICTION --------------------

print("\nCO5: FINAL SAMPLE PREDICTION")

sample_house = X_test.iloc[[0]]
estimated_price = regression_model.predict(sample_house)[0]
estimated_category = classification_model.predict(sample_house)[0]

print("\nSample house details:")
print(sample_house.to_string(index=False))
print("Estimated price (original dataset units):", round(estimated_price, 2))
print("Estimated category:", estimated_category)

print("\nAll output files saved in the outputs folder.")
print("PROJECT COMPLETED SUCCESSFULLY")
