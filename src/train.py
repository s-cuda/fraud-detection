import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys
import os
import time
sys.path.append(os.path.dirname(__file__))
from features import load_and_merge, build_all_features


def time_based_split(df: pd.DataFrame):
    """
    Split by time — not random.
    Train on first 80% of time, test on last 20%.
    This prevents data leakage — we never train on future data.
    """
    df = df.sort_values('TransactionDT')
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]
    print(f"Train: {len(train):,} rows  |  Test: {len(test):,} rows")
    print(f"Train fraud rate: {train['isFraud'].mean()*100:.2f}%")
    print(f"Test  fraud rate: {test['isFraud'].mean()*100:.2f}%")
    return train, test


def get_features_and_target(df: pd.DataFrame):
    drop_cols = ['TransactionID', 'isFraud', 'TransactionDT',
                 'card1', 'card2', 'card3', 'card5']
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df['isFraud']
    return X, y

def target_encode(train_df, test_df, cols, target='isFraud'):
    """
    For each column, replace category value with its fraud rate
    computed ONLY on training data, then applied to test.
    This prevents leakage — test fraud rates never touch training.
    """
    for col in cols:
        if col not in train_df.columns:
            continue
        fraud_rate = train_df.groupby(col)[target].mean()
        train_df[col] = train_df[col].map(fraud_rate)
        test_df[col]  = test_df[col].map(fraud_rate)
        # Unseen values in test get the global mean
        global_mean = train_df[target].mean()
        train_df[col] = train_df[col].fillna(global_mean)
        test_df[col]  = test_df[col].fillna(global_mean)
    return train_df, test_df

def train_baseline(X_train, y_train, X_test, y_test):
    """
    Logistic regression baseline.
    Gives us a benchmark AUC to beat with LightGBM.
    """
    string_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    print(f"Remaining string columns: {string_cols}")

    print("\n--- Baseline: Logistic Regression ---")
    # Fill missing for logistic regression
    X_tr = X_train.fillna(-999)
    X_te = X_test.fillna(-999)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr_scaled, y_train)

    preds = model.predict_proba(X_te_scaled)[:, 1]
    auc   = roc_auc_score(y_test, preds)
    print(f"Baseline AUC: {auc:.4f}")
    return auc


def train_lightgbm(X_train, y_train, X_test, y_test):
    """
    LightGBM with MLflow tracking.
    scale_pos_weight handles the 27:1 class imbalance.
    """
    print("\n--- LightGBM ---")

    # Class imbalance ratio
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos
    print(f"scale_pos_weight: {scale:.1f}")

    params = {
        'objective':         'binary',
        'metric':            'auc',
        'learning_rate':     0.05,
        'num_leaves':        128,       # was 64
        'max_depth':         -1,
        'min_child_samples': 50,        # was 100
        'feature_fraction':  0.8,
        'bagging_fraction':  0.8,
        'bagging_freq':      5,
        'scale_pos_weight':  scale,
        'random_state':      42,
        'verbose':           -1,
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest  = lgb.Dataset(X_test,  label=y_test, reference=dtrain)

    run_name = f"lgbm_lr{params['learning_rate']}_leaves{params['num_leaves']}_{int(time.time())}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True), 
            lgb.log_evaluation(period=100)
        ]

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dtest],
            callbacks=callbacks
        )

        preds = model.predict(X_test)
        auc   = roc_auc_score(y_test, preds)

        mlflow.log_metric("auc", auc)
        mlflow.log_metric("best_iteration", model.best_iteration)
        mlflow.lightgbm.log_model(model, name="model")

        print(f"\nLightGBM AUC : {auc:.4f}")
        print(f"Best iteration: {model.best_iteration}")

        # Threshold analysis
        print("\n--- Threshold Analysis ---")
        for threshold in [0.3, 0.4, 0.5, 0.6]:
            preds_binary = (preds >= threshold).astype(int)
            report = classification_report(
                y_test, preds_binary,
                target_names=['legit', 'fraud'],
                output_dict=True
            )
            fraud_precision = report['fraud']['precision']
            fraud_recall    = report['fraud']['recall']
            print(f"Threshold {threshold} → "
                  f"Precision: {fraud_precision:.3f}  "
                  f"Recall: {fraud_recall:.3f}")

    return model, auc

if __name__ == "__main__":
    BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TXN_PATH = os.path.join(BASE, "data", "train_transaction.csv")
    ID_PATH  = os.path.join(BASE, "data", "train_identity.csv")

    # Load and engineer features
    df = load_and_merge(TXN_PATH, ID_PATH)
    df = build_all_features(df)

    # Time based split
    train_df, test_df = time_based_split(df)

    # Target encode AFTER split — no leakage
    print("Target encoding high-cardinality columns...")
    encode_cols = ['P_emaildomain', 'R_emaildomain']
    train_df, test_df = target_encode(train_df, test_df, encode_cols)

    # Get features and target
    X_train, y_train = get_features_and_target(train_df)
    X_test,  y_test  = get_features_and_target(test_df)

    print(f"\nFeature count: {X_train.shape[1]}")

    RUN_BASELINE = False  # set True only when you want to rerun baseline

    if RUN_BASELINE:
        baseline_auc = train_baseline(X_train, y_train, X_test, y_test)
        print(f"Baseline AUC : {baseline_auc:.4f}")
    else:
        print("Baseline AUC : 0.8484 (skipped rerun)")
        baseline_auc = 0.8484

    # LightGBM
    model, lgbm_auc = train_lightgbm(X_train, y_train, X_test, y_test)

    print(f"\n{'='*40}")
    #print(f"Baseline AUC : {baseline_auc:.4f}")
    print(f"LightGBM AUC : {lgbm_auc:.4f}")
    #print(f"Improvement  : +{lgbm_auc - baseline_auc:.4f}")

    # Save model
    model_path = os.path.join(BASE, 'models', f'lgbm_auc{lgbm_auc:.4f}.txt')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")