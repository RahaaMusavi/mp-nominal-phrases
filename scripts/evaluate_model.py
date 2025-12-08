# scripts/evaluate_model.py

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Load preprocessing function
from scripts.preprocess_data import preprocess_conllu_folder

def evaluate_model(pipeline_path="artifacts/best_pipeline_combined.joblib", raw_data_path="data/raw_conllu"):
    # Step 1: Preprocess the raw data
    df = preprocess_conllu_folder(raw_data_path)
    print(f"Preprocessed {len(df)} rows of evaluation data.")

    # Step 2: Load label encoders
    le_combined = joblib.load("artifacts/le_combined.joblib")

    # Step 3: Prepare features and target
    drop_cols = ['nominal_head_form', 'modifier_form', 'ezafe_label', 'position', 'combined_label', 'ezafe_label_enc', 'combined_label_enc']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df['ezafe_label_enc'] = joblib.load("artifacts/le_ezafe.joblib").transform(df['ezafe_label'])
    df['combined_label'] = df['ezafe_label_enc'].astype(str) + "_" + df['position'].astype(str)
    y_comb = le_combined.transform(df['combined_label'])

    # Step 4: Load trained pipeline
    pipeline = joblib.load(pipeline_path)

    # Step 5: Stratified k-fold evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y_comb), start=1):
        X_test = X.iloc[test_idx]
        y_test = y_comb[test_idx]

        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_w = f1_score(y_test, y_pred, average='weighted')
        fold_results.append({'fold': fold_idx, 'accuracy': acc, 'f1_weighted': f1_w})
        print(f"Fold {fold_idx} - Accuracy: {acc:.4f}, F1-weighted: {f1_w:.4f}")

    # Step 6: Overall evaluation
    y_pred_all = pipeline.predict(X)
    print("\nClassification Report (all data):")
    print(classification_report(y_comb, y_pred_all, zero_division=0))
    print("Confusion Matrix (all data):")
    print(confusion_matrix(y_comb, y_pred_all))

    # Save evaluation results
    eval_df = pd.DataFrame(fold_results)
    eval_df.to_csv("artifacts/evaluation_results.csv", index=False)
    print("Saved fold-wise evaluation results to artifacts/evaluation_results.csv")

if __name__ == "__main__":
    evaluate_model()