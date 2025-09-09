#-------------------------------------------------------------------------------
# Name:        predict.py
# Purpose:
#
# Author:      bluer
#
# Created:     20/08/2025
# Copyright:   (c) bluer 2025
# Licence:     MIT
#-------------------------------------------------------------------------------
import pandas as pd
import joblib
import os
import sys

MODEL_PATH = "trained_fungi_model.pkl"

def load_model(path=MODEL_PATH):
     """
    Load the pretrained fungi model pipeline.

    Parameters:
        path (str): Path to the saved pipeline file.

    Returns:
        clf_pipeline: scikit-learn pipeline
    """
    if not os.path.exists(path):
        sys.exit(f"ERROR: Model file not found at '{path}'."
                f"Please run training or update MODEL_PATH.")
    clf_pipeline = joblib.load(path)
    print(f"Loaded pretrained model from '{path}'")
    return clf_pipeline

def predict(samples_csv, output_csv=None):
    """
    Make predictions on new samples CSV.

    Parameters:
        samples_csv (str): Path to CSV with new sample features.
        output_csv (str, optional): Path to save predictions. Defaults to None.

    Returns:
        y_pred (pd.Series): Predicted dominant fungi
    """
    if not os.path.exists(samples_csv):
        sys.exit(f"ERROR: Input CSV not found at '{samples_csv}'.")

    clf_pipeline = load_model()
    X_new = pd.read_csv(samples_csv)

    #Check that required features exist
    model_features = clf_pipeline.named_steps['preprocessor'].transformers_[0][2] + \
        list(clf_pipline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out())
    missing_features = [f for f in X_new.columns if f not in model_features]
    if missing_features:
        print(f"Warning: Input CSV has columns not used in model: '{missing_features}'")

    y_pred = clf_pipeline.predict(X_new)

    if output_csv:
        pd.DataFrame({'dominant_fungus': y_pred}).to_csv(output_csv, index=False)
        print(f"Predictions save to '{output_csv}'.")

    return pd.Series(y_pred, name= 'dominant_fungus')

if _name_ == "_main_":
    import argparse

    parser = argparse.ArguementParser(description="Predict dominant fungi from sample CSV.")
    parser.add_arguement("samples_csv", help="Path to CSV containing new sample features")
    parser.add_arguement("--output", help="Path to save predictions CSV", default=None)
    args = parser.parse_args()

    predictions = predict(args.samples_csv, args.output)
    print("\nPredicted dominant fungi:")
    print(predictions)