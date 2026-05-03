"""
Transaction replay script for the fraud detection dashboard.

Loads the test split of the IEEE-CIS dataset, engineers features using the
same pipeline as training, then sends each transaction to the running API
one at a time with a configurable delay.

The dashboard live feed updates as transactions arrive. Each request includes
display_card and display_email so the feed shows readable names instead of
encoded numbers, and actual_label so the feed can show whether the model
was correct.

Usage:
    python src/replay.py
    python src/replay.py --delay 0.5

Requirements:
    API must be running: docker-compose up
"""

import pandas as pd
import numpy as np
import requests
import time
import sys
import os
import argparse

sys.path.append(os.path.dirname(__file__))
from features import load_and_merge, build_all_features

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXN_PATH = os.path.join(BASE, 'data', 'train_transaction.csv')
ID_PATH  = os.path.join(BASE, 'data', 'train_identity.csv')
API_URL  = 'http://127.0.0.1:8000/predict'

# These columns stay as strings when building the API payload.
# All other feature columns are cast to float.
STRING_COLS = {'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain'}

# Columns dropped before sending to the API.
DROP_COLS = ['TransactionID', 'isFraud', 'TransactionDT',
             'card1', 'card2', 'card3', 'card5']


def run_replay(delay_seconds: float = 1.5, max_transactions: int = None):
    print("Loading data...")

    # Load raw transaction CSV first to capture original string values for
    # card network and email domain BEFORE feature engineering encodes them
    # into numeric label codes. We need these for the dashboard display.
    raw_txn   = pd.read_csv(TXN_PATH, usecols=['TransactionID', 'card4', 'P_emaildomain'])
    raw_cards  = raw_txn.set_index('TransactionID')['card4'].fillna('unknown').astype(str)
    raw_emails = raw_txn.set_index('TransactionID')['P_emaildomain'].fillna('unknown').astype(str)
    del raw_txn

    print("Engineering features...")
    df = load_and_merge(TXN_PATH, ID_PATH)

    # Store TransactionIDs before feature engineering drops them
    transaction_ids = df['TransactionID'].values.copy()

    df = build_all_features(df)
    df = df.sort_values('TransactionDT')

    # Re-align transaction IDs to match the sorted order
    txn_id_series = pd.Series(transaction_ids, index=range(len(transaction_ids)))

    split_idx = int(len(df) * 0.8)
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()

    # Apply target encoding on email domains using training data only.
    # This matches what happens during model training to avoid leakage.
    for col in ['P_emaildomain', 'R_emaildomain']:
        fraud_rate    = train_df.groupby(col)['isFraud'].mean()
        global_mean   = train_df['isFraud'].mean()
        test_df[col]  = test_df[col].map(fraud_rate).fillna(global_mean)

    labels    = test_df['isFraud'].values
    drop_cols = [c for c in DROP_COLS if c in test_df.columns]
    features  = test_df.drop(columns=drop_cols)

    # Get the original integer positions in the sorted DataFrame for the test split.
    # These map back to the original TransactionIDs so we can look up display values.
    test_original_positions = df.index[split_idx:]

    print(f"Replaying {len(features):,} test transactions...")
    print(f"Actual fraud rate in test set: {labels.mean()*100:.2f}%")
    print(f"Delay between transactions: {delay_seconds}s")
    print("Press Ctrl+C to stop\n")

    total = flagged = correct = 0

    for i, (idx, row) in enumerate(features.iterrows()):
        if max_transactions and i >= max_transactions:
            break

        # Build payload — numeric features as float, strings as str, missing as None
        payload = {}
        for k, v in row.items():
            if pd.isna(v):
                payload[k] = None
            elif k in STRING_COLS:
                payload[k] = str(v)
            else:
                payload[k] = float(v)

        # Look up the original TransactionID for this row using its position
        # in the sorted DataFrame, then use it to fetch the original card and email strings.
        original_pos = test_original_positions[i]
        txn_id       = txn_id_series.iloc[original_pos] if original_pos < len(txn_id_series) else None

        if txn_id is not None and txn_id in raw_cards.index:
            card_display  = raw_cards[txn_id]
            email_display = raw_emails.get(txn_id, 'unknown')
        else:
            card_display  = 'unknown'
            email_display = 'unknown'

        payload['display_card']  = card_display  if card_display  != 'nan' else 'unknown'
        payload['display_email'] = email_display if email_display != 'nan' else 'unknown'
        payload['actual_label']  = int(labels[i])

        try:
            r = requests.post(API_URL, json=payload, timeout=10)

            if r.status_code == 200:
                result     = r.json()
                score      = result['fraud_score']
                actual     = int(labels[i])
                predicted  = 1 if score >= 0.4 else 0
                is_correct = predicted == actual

                total   += 1
                flagged += predicted
                correct += int(is_correct)

                if score >= 0.7:
                    colour = '\033[91m'
                elif score >= 0.35:
                    colour = '\033[93m'
                else:
                    colour = '\033[92m'
                reset = '\033[0m'

                print(
                    f"Txn {total:>5} | "
                    f"Actual: {'FRAUD' if actual else 'legit':>5} | "
                    f"Score: {colour}{score:.3f}{reset} | "
                    f"{'correct' if is_correct else 'wrong':>7} | "
                    f"Latency: {result['latency_ms']:.0f}ms | "
                    f"{card_display} · {email_display}"
                )

                if total % 50 == 0:
                    print(f"\n  After {total} transactions:")
                    print(f"  Flagged as fraud : {flagged} ({flagged/total*100:.1f}%)")
                    print(f"  Correct          : {correct} ({correct/total*100:.1f}%)")
                    print(f"  Actual fraud     : {labels[:total].sum()} ({labels[:total].mean()*100:.1f}%)\n")

            else:
                print(f"API error {r.status_code}: {r.text[:120]}")

        except Exception as e:
            print(f"Request failed: {e}")

        time.sleep(delay_seconds)

    print(f"\nReplay complete. {total} transactions processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay test transactions through the fraud detection API")
    parser.add_argument('--delay', type=float, default=1.5, help='Seconds between transactions (default 1.5)')
    parser.add_argument('--max',   type=int,   default=None, help='Maximum transactions to replay')
    args = parser.parse_args()
    run_replay(delay_seconds=args.delay, max_transactions=args.max)