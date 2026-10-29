
# ============================================================
# MINI PROJECT: HOUSE PRICE ANALYSIS AND PREDICTION
# Subject: Python Programming for Data Science
# CO1: NumPy and Pandas
# CO2: Data Preprocessing
# CO3: Data Visualization
# CO4: Regression and Classification
# CO5: Model Development and Evaluation
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# PROJECT SETTINGS
# ------------------------------------------------------------

DATA_FILE = os.path.join("data", "Pune_House_Data.csv")
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("       HOUSE PRICE ANALYSIS AND PREDICTION PROJECT")
print("=" * 65)

# ============================================================
# CO1: NUMPY AND PANDAS OPERATIONS
# ============================================================

print("\nCO1: NUMPY AND PANDAS OPERATIONS")

# NumPy array operations
sample = np.array([10, 20, 30, 40, 50])

print("\nNumPy array:", sample)
print("First element:", sample[0])
print("First three elements:", sample[:3])
print("Reshaped array:\n", sample.reshape(5, 1))
print("Adding 5 to each element:", sample + 5)
print("Mean:", np.mean(sample))
print("Median:", np.median(sample))
print("Standard deviation:", np.std(sample))

# Load real-world housing dataset
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Place Pune_House_Data.csv inside the data folder."
    )

df = pd.read_csv(DATA_FILE)

print("\nOriginal dataset shape:", df.shape)
print("\nFirst five rows:\n", df.head())
print("\nColumn names:", df.columns.tolist())
print("\nDataset information:")
df.info()
print("\nStatistical summary:\n", df.describe(include="all"))

# ============================================================
# CO2: DATA PREPROCESSING
# ============================================================

print("\n\nCO2: DATA PREPROCESSING")

# Remove exact duplicate rows
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates().copy()

print("Duplicate rows removed:", duplicate_count)

# Convert numeric columns safely
for column in ["bath", "balcony", "price"]:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

# Extract number of bedrooms from values such as 2 BHK
if "size" in df.columns:
    df["bhk"] = pd.to_numeric(
        df["size"].astype("string").str.extract(r"(\d+)")[0],
        errors="coerce"
    )

# Convert square-foot measurements to numeric values.
# For ranges, use the midpoint. Unrecognized values become missing.
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

# Keep valid positive target values
df = df[df["price"].notna() & (df["price"] > 0)].copy()

# Remove clearly invalid area or bedroom values
if "total_sqft" in df.columns:
    df.loc[df["total_sqft"] <= 0, "total_sqft"] = np.nan

if "bhk" in df.columns:
    df.loc[df["bhk"] <= 0, "bhk"] = np.nan

print("Cleaned dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows remaining:", df.duplicated().sum())

# Filtering and sorting
print("\nProperties with at least 2 bedrooms:")
print(df[df["bhk"] >= 2].head())

print("\nFive highest recorded prices:")
print(df.sort_values("price", ascending=False)[
    ["site_location", "bhk", "total_sqft", "price"]
].head())

# Grouping and aggregation
print("\nAverage price by location (top 10):")
location_summary = (
    df.groupby("site_location")["price"]
    .agg(["mean", "median", "count"])
    .sort_values("mean", ascending=False)
)

print(location_summary.head(10))

location_summary.to_csv(
    os.path.join(OUTPUT_DIR, "location_summary.csv")
)

# Save the cleaned dataset
cleaned_file = os.path.join(OUTPUT_DIR, "cleaned_housing_data.csv")
df.to_csv(cleaned_file, index=False)

# ============================================================
# CO3: DATA VISUALIZATION USING MATPLOTLIB AND SEABORN
# ============================================================

print("\n\nCO3: DATA VISUALIZATION")

sns.set_theme(style="whitegrid")

# 1. Price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=40, kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Price (dataset units)")
plt.ylabel("Number of Properties")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "price_distribution.png"))
plt.show()
plt.close()

# 2. Number of bedrooms
plt.figure(figsize=(8, 5))
df["bhk"].value_counts().sort_index().head(10).plot(kind="bar")
plt.title("Number of Properties by Bedroom Count")
plt.xlabel("Bedrooms (BHK)")
plt.ylabel("Number of Properties")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "bedroom_counts.png"))
plt.show()
plt.close()

# 3. Area versus price
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="total_sqft",
    y="price",
    alpha=0.5
)
plt.title("Property Area versus Price")
plt.xlabel("Total Area (square feet)")
plt.ylabel("Price (dataset units)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "area_vs_price.png"))
plt.show()
plt.close()

# 4. Correlation heatmap
numeric_data = df.select_dtypes(include=np.number)

plt.figure(figsize=(8, 6))
sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.show()
plt.close()

# ============================================================
# CO4: MACHINE LEARNING
# REGRESSION AND CLASSIFICATION
# ============================================================

print("\n\nCO4: MACHINE LEARNING")

# Select available predictive features
numeric_candidates = ["total_sqft", "bath", "balcony", "bhk"]
categorical_candidates = ["site_location", "area_type", "availability"]

numeric_features = [
    col for col in numeric_candidates if col in df.columns
]
categorical_features = [
    col for col in categorical_candidates if col in df.columns
]

feature_columns = numeric_features + categorical_features

if not numeric_features:
    raise ValueError("No suitable numeric input features were found.")

# Use the recorded price as the regression target
X = df[feature_columns].copy()
y = df["price"].copy()

# Exclude rows with missing values in the target
valid_target = y.notna() & (y > 0)
X = X.loc[valid_target]
y = y.loc[valid_target]

if len(X) < 20:
    raise ValueError("Not enough valid rows to train and evaluate the models.")

# Create train and test sets before fitting preprocessing steps
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Input features:", feature_columns)

# Preprocessing pipelines learn from training data only
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []

if numeric_features:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(transformers=transformers)

# ------------------------------------------------------------
# MODEL 1: LINEAR REGRESSION
# ------------------------------------------------------------

regression_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LinearRegression())
])

regression_model.fit(X_train, y_train)

predictions = regression_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nLINEAR REGRESSION RESULTS")
print("Mean Absolute Error (MAE):", round(mae, 3))
print("Root Mean Squared Error (RMSE):", round(rmse, 3))
print("R-squared (R2):", round(r2, 3))

# Compare actual and predicted values
comparison = pd.DataFrame({
    "Actual Price": y_test.to_numpy(),
    "Predicted Price": predictions
})

print("\nActual versus predicted prices:")
print(comparison.head(10))

comparison.to_csv(
    os.path.join(OUTPUT_DIR, "prediction_comparison.csv"),
    index=False
)

# Regression plot
plt.figure(figsize=(7, 5))
plt.scatter(y_test, predictions, alpha=0.5)
minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())
plt.plot([minimum, maximum], [minimum, maximum], "r--")
plt.title("Actual versus Predicted House Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "regression_results.png"))
plt.show()
plt.close()

# ------------------------------------------------------------
# MODEL 2: LOGISTIC REGRESSION CLASSIFICATION
# ------------------------------------------------------------

# Derive price categories from the training target only.
# These categories are relative to this dataset, not official
# real-estate market categories.

lower_threshold = y_train.quantile(1 / 3)
upper_threshold = y_train.quantile(2 / 3)

def price_category(values):
    return np.where(
        values <= lower_threshold,
        "Budget",
        np.where(values <= upper_threshold, "Mid-Range", "Premium")
    )

y_train_class = price_category(y_train)
y_test_class = price_category(y_test)

classification_model = Pipeline([
    ("preprocessing", ColumnTransformer(transformers=transformers)),
    ("model", LogisticRegression(max_iter=2000))
])

classification_model.fit(X_train, y_train_class)
class_predictions = classification_model.predict(X_test)

print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", round(
    accuracy_score(y_test_class, class_predictions), 3
))

print("\nClassification Report:")
print(classification_report(
    y_test_class,
    class_predictions,
    zero_division=0
))

ConfusionMatrixDisplay.from_predictions(
    y_test_class,
    class_predictions,
    labels=["Budget", "Mid-Range", "Premium"],
    cmap="Blues"
)
plt.title("Price Category Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()
plt.close()

# ============================================================
# CO5: COMPLETE SOLUTION AND SAMPLE PREDICTION
# ============================================================

print("\n\nCO5: PROJECT EVALUATION AND SAMPLE PREDICTION")

# Predict using a real example from the held-out test set
sample_house = X_test.iloc[[0]]

estimated_price = regression_model.predict(sample_house)[0]
estimated_category = classification_model.predict(sample_house)[0]

print("\nSample house input:")
print(sample_house.to_string(index=False))

print("\nEstimated price (dataset units):", round(estimated_price, 2))
print("Predicted price category:", estimated_category)

print("\nAll generated files are stored in:", OUTPUT_DIR)

print("\n" + "=" * 65)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 65)
