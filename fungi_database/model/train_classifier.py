#-------------------------------------------------------------------------------
# Name:        fungi_model_train_test
# Purpose:
#
# Author:      bluer
#
# Created:     19/08/2025
# Copyright:   (c) bluer 2025
# Licence:     MIT
#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.impute import SimpleImputer

#Load cleaned dataset
df = pd.read_csv(r"fungi_clean.csv", low_memory=False)

#Identify species abundance cols
metaData_cols = ["sample_ID", "paper_ID", "paper_title", "paper_year", "paper_authors", "paper_journal",
    "paper_doi", "paper_sample_name", "latitude", "longitude", "elevation_study", "continent", "country",
    "location", "year_of_sampling_from", "year_of_sampling_to", "month_of_sampling", "day_of_sampling",
    "sample_type", "sample_type_specification", "environment_type", "ecosystem_classification", "dominant_plant_species",
    "other_plant_species", "manipulated", "experimental_manipulation_type", "experimental_manipulation_direction",
     "experimental_manipulation_vegetation", "experimental_manipulation_duration", "experimental_manipulation_frequency",
     "experimental_manipulation_application_detail", "experimental_manipulation_intensity", "pH", "pH_method",
     "organic_matter_content", "organic_C_content", "total_N_content", "total_Ca", "total_P", "total_K", "MAT_study",
     "MAP_study", "area_GPS", "area_sampled", "number_of_subsamples", "sampling_info", "sample_depth", "sample_info",
     "DNA_extraction_sample_mass", "DNA_extraction_size", "DNA_extraction_method", "barcoding_region", "PCR_primers",
     "PCR_primers_sequence", "sequencing_platform", "ITS1_extracted", "ITS2_extracted", "ITS_total", "date_added",
     "submitted_by", "flag_outliers"]
fungi_cols = [c for c in df.columns if c not in metaData_cols]

#Check for invalid feature values
non_numeric_cols = [c for c in fungi_cols if not np.issubdtype(df[c].dtype, np.number)]
if non_numeric_cols:
    raise SystemExit(
        f"Error: Non-numeric values detected in fungal abundance columns: {non_numeric_cols}\nPlease re-run preprocessing before training."
    )
else:
    print("All fungal abundance columns numeric.")

#Derive dominant fungus
df["dominant_fungus"] = df[fungi_cols].idxmax(axis=1)
df["max_val"] = df[fungi_cols].max(axis=1)
df["tie_flag"] = df[fungi_cols].eq(df["max_val"], axis=0).sum(axis=1) > 1
print("tie_flag")

#Define X (features) and y (target)
chosen_cols = ["dominant_plant_species", "pH", "organic_content",
                "MAT_study", "MAP_study", "elevation_study",
                "total_K", "total_P", "total_Ca", "total_N_content",
                "organic_C_content", "ecosystem_classification"]
feature_cols = [c for c in chosen_cols if c in df.columns]
missing_cols = [c for c in chosen_cols if c not in df.columns]
if missing_cols:
    print(f"Warning: these feature columns are missing and will be skipped: {missing_cols}")

X = df[feature_cols]
y = df["dominant_fungus"]

#Rare cases isolated
counts = y.value_counts()
rare_classes = counts[counts == 1].index
if len(rare_classes) > 0:
    print(f"Grouping {len(rare_classes)} rare classes into 'Other'")
    y = y.replace(rare_classes, "Other")

counts = y.value_counts()
too_small = counts[counts <2].index
if len(too_small) >0:
    print(f"Warning: classes with <2 samples found ({list(too_small)}). Grouping as other.")
    y = y.replace(too_small, "Other")

#Identify numeric and categorical features
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

#Column transform with imputers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),   # fill numeric NaNs with mean
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),  # fill categorical NaNs with mode
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
        ]), categorical_features),
    ]
)

#Create pipeline
clf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])
#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#Train pipeline
clf_pipeline.fit(X_train, y_train)

#Eval
y_pred = clf_pipeline.predict(X_test)
print("Classification report:")
print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred, zero_division=0)
with open("classification_report.txt", "w") as f:
    f.write(report)
print("Classification report saved to classification_report.txt")

# Save as CSV with all classes included
all_classes = sorted(y_test.unique())
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=all_classes, zero_division=0
)
df_report = pd.DataFrame({
    "class": all_classes,
    "precision": precision,
    "recall": recall,
    "f1-score": f1,
    "support": support
})
df_report.to_csv("classification_report.csv", index=False)
print("Classification report saved to classification_report.csv")

#Dump to joblib for use without training
joblib.dump(clf_pipeline, "trained_fungi_model.pkl")
print(f"Model saved to 'trained_fungi_model.pkl.'")
