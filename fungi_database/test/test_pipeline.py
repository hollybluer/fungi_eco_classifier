#-------------------------------------------------------------------------------
# Name:        test_pipeline
# Purpose:
#
# Author:      bluer
#
# Created:     21/08/2025
# Copyright:   (c) bluer 2025
# Licence:     MIT
#-------------------------------------------------------------------------------
import os
import joblib
import pandas as pd
import numpy as np
import pytest

from scikitlearn import train_test_split
from scikitlearn import pipeline
from scikitlearn import RandomForestClassifier
from scikitlearn import OneHotEncoder
from scikitlearn import ColumnTransformer

@pytest.fixture
def sample_data():
    df = pd.DataFrame({
    "dominant_plant_species": ["oak", "pine", "oak", "spruce", "pine", "oak"],
    "pH": [5.2, 6.1, 5.8, 7.0, 5.5, 6.3],
    "organic_C_content": [1.2, 2.5, 1.8, 3.1, 2.0, 2.2],
    "ecosystem_classification": ["forest", "forest", "grassland", "forest", "wetland", "forest"],
    "dominant_fungus": ["fungus_a", "fungus_b", "fungus_a", "fungus_b", "fungus_a", "fungus_b"]
})
return df

def build_pipline(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.selct_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    "classifier", RandomForestClassifier(n_estimators=10, random_state=42)
    ])

    return pipeline

def test_pipeline_fit_predict(tmp_path, sample_data):
    X = sample_data.drop(columns=["dominant_fungus"])
    y = sample_data["dominant_fungus"]

    clf = build_pipeline(X)
    clf.fit(X,y)
    preds = clf.predict(X)

    assert len(preds) ==len(y)

    model_path = tmp_path / "test_model.pkl"
    joblib.dump(clf, model_path)

    loaded = joblib.load(model_path)
    preds2 = loaded.predict(X)
    assert len(preds2) == len(y)
