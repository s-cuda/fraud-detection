"""
Drift detection using Evidently AI 0.7.x

Reference = training data (first 80% by time) — what the model learned from.
Current   = test data   (last 20% by time)   — what arrived after deployment.

Saves an HTML report and a JSON summary consumed by the API dashboard.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(__file__))
from features import load_and_merge, build_all_features

from evidently import Report
from evidently.presets import DataDriftPreset

BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXN_PATH     = os.path.join(BASE, 'data', 'train_transaction.csv')
ID_PATH      = os.path.join(BASE, 'data', 'train_identity.csv')
REPORT_PATH  = os.path.join(BASE, 'models', 'drift_report.html')
SUMMARY_PATH = os.path.join(BASE, 'models', 'drift_summary.json')

# Monitor interpretable features only
# V columns are anonymized so we exclude them
MONITOR_FEATURES = [
    'TransactionAmt', 'amt_log', 'amt_to_card_mean_ratio',
    'amt_z_score_card', 'hour', 'day_of_week', 'is_night',
    'card_age', 'is_new_card', 'card_txn_count',
    'card_unique_addr', 'card_unique_email', 'card_amt_std',
    'card_time_since_first', 'uid_txn_count', 'uid_amt_mean',
    'P_emaildomain', 'R_emaildomain',
    'P_is_free_email', 'R_is_free_email',
]


def load_and_split():
    print("Loading data...")
    df = load_and_merge(TXN_PATH, ID_PATH)
    df = build_all_features(df)
    df = df.sort_values('TransactionDT')

    split_idx = int(len(df) * 0.8)
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()

    # Target encode email domains — same as training pipeline
    for col in ['P_emaildomain', 'R_emaildomain']:
        fraud_rate    = train_df.groupby(col)['isFraud'].mean()
        global_mean   = train_df['isFraud'].mean()
        train_df[col] = train_df[col].map(fraud_rate).fillna(global_mean)
        test_df[col]  = test_df[col].map(fraud_rate).fillna(global_mean)

    print(f"Reference (train): {len(train_df):,} rows")
    print(f"Current   (test) : {len(test_df):,} rows")
    return train_df, test_df


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    # Keep only monitored features that exist in both datasets
    available = [f for f in MONITOR_FEATURES
                 if f in reference.columns and f in current.columns]
    print(f"Running drift detection on {len(available)} features...")

    ref = reference[available].copy()
    cur = current[available].copy()

    # Sample to speed up — 50k rows is more than enough for stable statistics
    if len(ref) > 50000:
        ref = ref.sample(50000, random_state=42)
    if len(cur) > 20000:
        cur = cur.sample(20000, random_state=42)

    print(f"Running Evidently on {len(ref):,} reference rows and {len(cur):,} current rows...")

    report = Report([DataDriftPreset()])
    result = report.run(ref, cur)

    # Save HTML report
    result.save_html(REPORT_PATH)
    print(f"HTML report saved → {REPORT_PATH}")

    # Extract summary from result dict
    result_dict = result.dict()

    # Navigate the Evidently 0.7.x result structure
    feature_results = []
    dataset_drifted = False
    features_drifted = 0

    try:
        # Evidently 0.7 stores results in a nested structure
        metrics = result_dict.get('metrics', [])
        for metric in metrics:
            metric_id = metric.get('metric_id', '')
            value     = metric.get('value', {})

            # Dataset-level drift
            if 'DatasetDrift' in metric_id or 'dataset_drift' in str(value).lower():
                dataset_drifted  = value.get('dataset_drift', False)
                features_drifted = value.get('number_of_drifted_columns', 0)

            # Per-column drift
            if 'ColumnDrift' in metric_id or 'column_name' in str(value):
                col_name = value.get('column_name', '')
                if col_name in available:
                    feature_results.append({
                        'feature':   col_name,
                        'drifted':   bool(value.get('drift_detected', False)),
                        'statistic': round(float(value.get('statistic', 0)), 4),
                        'p_value':   round(float(value.get('p_value', 1.0)), 4),
                    })

    except Exception as e:
        print(f"Warning: Could not parse detailed results — {e}")
        print("Computing drift manually using KS test...")
        feature_results = compute_ks_drift(ref, cur, available)
        features_drifted = sum(1 for f in feature_results if f['drifted'])
        dataset_drifted  = features_drifted > len(available) * 0.2

    if not feature_results:
        print("Falling back to manual KS test...")
        feature_results = compute_ks_drift(ref, cur, available)
        features_drifted = sum(1 for f in feature_results if f['drifted'])
        dataset_drifted  = features_drifted > len(available) * 0.2

    feature_results.sort(key=lambda x: x['statistic'], reverse=True)

    summary = {
        'checked_at':       pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'reference_size':   int(len(reference)),
        'current_size':     int(len(current)),
        'features_checked': len(available),
        'features_drifted': int(features_drifted),
        'drift_share':      round(features_drifted / max(len(available), 1), 3),
        'dataset_drift':    bool(dataset_drifted),
        'feature_results':  feature_results,
    }

    with open(SUMMARY_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {SUMMARY_PATH}")

    return summary


def compute_ks_drift(ref, cur, available, threshold=0.05):
    from scipy import stats
    results = []
    for feat in available:
        try:
            r = ref[feat].dropna().values
            c = cur[feat].dropna().values
            if len(r) < 10 or len(c) < 10:
                continue
            ks_stat, p_value = stats.ks_2samp(r, c)
            # Use both p-value AND KS stat > 0.05 to avoid
            # flagging trivially small differences in large samples
            results.append({
                'feature':   feat,
                'drifted':   bool(p_value < threshold and ks_stat > 0.05),
                'statistic': round(float(ks_stat), 4),
                'p_value':   round(float(p_value), 4),
            })
        except Exception:
            continue
    return results


def print_summary(summary: dict):
    print("\n" + "="*60)
    print("DRIFT DETECTION SUMMARY")
    print("="*60)
    print(f"Checked at      : {summary['checked_at']}")
    print(f"Reference rows  : {summary['reference_size']:,}")
    print(f"Current rows    : {summary['current_size']:,}")
    print(f"Features checked: {summary['features_checked']}")
    print(f"Features drifted: {summary['features_drifted']}")
    print(f"Drift share     : {summary['drift_share']*100:.1f}%")
    print(f"Dataset drift   : {'YES ⚠' if summary['dataset_drift'] else 'NO ✓'}")
    print()
    print(f"{'':2} {'Feature':<28} {'Status':<10} {'KS stat':<10} {'p-value'}")
    print("-"*64)
    for f in summary['feature_results']:
        icon   = "⚠" if f['drifted'] else "✓"
        status = "DRIFTED" if f['drifted'] else "stable"
        print(f"{icon}  {f['feature']:<28} {status:<10} {f['statistic']:<10} {f['p_value']}")


if __name__ == "__main__":
    reference, current = load_and_split()
    summary = run_drift_report(reference, current)
    print_summary(summary)
    print(f"\nOpen in browser: {REPORT_PATH}")