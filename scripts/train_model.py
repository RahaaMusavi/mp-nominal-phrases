# scripts/train_model.py

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from boruta import BorutaPy
from sklearn.base import BaseEstimator, TransformerMixin, clone

# Import the preprocessing function
from scripts.preprocess_data import preprocess_conllu_folder

# -----------------------
# Boruta selector wrapper
# -----------------------

class BorutaSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator=None, n_estimators='auto', verbose=0, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        base_est = clone(self.estimator) if self.estimator else RandomForestClassifier(n_jobs=-1, random_state=self.random_state)
        self.boruta_ = BorutaPy(base_est, n_estimators=self.n_estimators, verbose=self.verbose, random_state=self.random_state)
        self.boruta_.fit(X_arr, y_arr)
        self.support_ = self.boruta_.support_.copy()
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        if not hasattr(self, "support_") or np.sum(self.support_) == 0:
            return X_arr
        return X_arr[:, self.support_]

    def get_support(self):
        return getattr(self, "support_", None)

# -----------------------
# Main training script
# -----------------------

def main():
    # Step 1: Preprocess the raw data
    df = preprocess_conllu_folder("data/raw_conllu")
    print(f"Preprocessed {len(df)} rows of data.")

    # Step 2: Encode labels
    le_ezafe = LabelEncoder()
    df['ezafe_label_enc'] = le_ezafe.fit_transform(df['ezafe_label'])

    df['combined_label'] = df['ezafe_label_enc'].astype(str) + "_" + df['position'].astype(str)
    le_combined = LabelEncoder()
    df['combined_label_enc'] = le_combined.fit_transform(df['combined_label'])

    # Save label encoders
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(le_ezafe, "artifacts/le_ezafe.joblib")
    joblib.dump(le_combined, "artifacts/le_combined.joblib")

    # Step 3: Prepare features and target
    drop_cols = ['nominal_head_form', 'modifier_form', 'ezafe_label', 'position', 'combined_label', 'ezafe_label_enc', 'combined_label_enc']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y_comb = df['combined_label_enc'].values

    # Step 4: ColumnTransformer for categorical features
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ], remainder='passthrough')

    # Step 5: Pipeline: Oversampling + Boruta + Random Forest
    boruta_base = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
    pipeline = ImbPipeline([
        ('pre', preprocessor),
        ('ros', RandomOverSampler(random_state=42)),
        ('boruta', BorutaSelector(estimator=boruta_base, n_estimators='auto', verbose=0, random_state=42)),
        ('clf', RandomForestClassifier(n_jobs=-1, random_state=42))
    ])

    # Step 6: Hyperparameter tuning
    param_dist = {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=10, cv=cv, scoring='f1_weighted', random_state=42, n_jobs=-1, verbose=2)
    random_search.fit(X, y_comb)
    print("RandomizedSearchCV complete. Best params:", random_search.best_params_)

    # Step 7: Save the best pipeline and selected features
    best_pipeline = random_search.best_estimator_
    joblib.dump(best_pipeline, "artifacts/best_pipeline_combined.joblib")

    pre_feature_names = best_pipeline.named_steps['pre'].get_feature_names_out()
    boruta_mask = best_pipeline.named_steps['boruta'].get_support()
    selected_feature_names = list(np.array(pre_feature_names)[boruta_mask]) if boruta_mask is not None else list(pre_feature_names)
    joblib.dump(selected_feature_names, "artifacts/selected_feature_names.joblib")
    print("Saved artifacts: best pipeline, selected features.")

if __name__ == "__main__":
    main()