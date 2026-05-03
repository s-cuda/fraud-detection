import json
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(__file__))

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXN_PATH = os.path.join(BASE, 'data', 'train_transaction.csv')
ID_PATH  = os.path.join(BASE, 'data', 'train_identity.csv')
OUT_PATH = os.path.join(BASE, 'models', 'analytics.json')

print("Loading data...")
train_txn = pd.read_csv(TXN_PATH)
train_id  = pd.read_csv(ID_PATH)
train     = train_txn.merge(train_id, on='TransactionID', how='left')

train['hour']     = (train['TransactionDT'] / 3600).astype(int) % 24
train['card_age'] = train['D1'].fillna(-1)

fraud = train[train['isFraud'] == 1]
legit = train[train['isFraud'] == 0]

print("Computing analytics...")

fraud_by_hour = {str(k): round(float(v), 4) for k, v in train.groupby('hour')['isFraud'].mean().items()}
vol_by_hour   = {str(k): int(v) for k, v in train.groupby('hour').size().items()}

card4_stats   = train.groupby('card4')['isFraud'].agg(['mean','count'])
fraud_by_card4 = {
    str(k): {'rate': round(float(v['mean']), 4), 'count': int(v['count'])}
    for k, v in card4_stats.iterrows() if v['count'] > 100
}

domain_stats   = train.groupby('P_emaildomain')['isFraud'].agg(['mean','count'])
domain_stats   = domain_stats[domain_stats['count'] > 200].sort_values('mean', ascending=False).head(10)
fraud_by_domain = {
    str(k): {'rate': round(float(v['mean']), 4), 'count': int(v['count'])}
    for k, v in domain_stats.iterrows()
}

fraud_amounts = fraud['TransactionAmt'].sample(500, random_state=42).round(2).tolist()
legit_amounts = legit['TransactionAmt'].sample(500, random_state=42).round(2).tolist()
fraud_ages    = fraud['card_age'][fraud['card_age'] >= 0].sample(500, random_state=42).round(1).tolist()
legit_ages    = legit['card_age'][legit['card_age'] >= 0].sample(500, random_state=42).round(1).tolist()

shap_importance = [
    {'feature': 'C13',                   'description': 'Billing address count',         'value': 0.224},
    {'feature': 'V70',                   'description': 'Vesta risk score (device)',      'value': 0.166},
    {'feature': 'C14',                   'description': 'Address count variant',          'value': 0.154},
    {'feature': 'card6',                 'description': 'Card type (credit/debit)',       'value': 0.129},
    {'feature': 'TransactionAmt',        'description': 'Transaction amount',             'value': 0.129},
    {'feature': 'card_txn_count',        'description': 'Card transaction count',         'value': 0.114},
    {'feature': 'C1',                    'description': 'Address count (C1)',             'value': 0.110},
    {'feature': 'R_emaildomain',         'description': 'Recipient email domain rate',    'value': 0.109},
    {'feature': 'card_time_since_first', 'description': 'Time since card first seen',     'value': 0.101},
    {'feature': 'D1',                    'description': 'Days since card first used',     'value': 0.101},
]

summary = {
    'total_transactions': int(len(train)),
    'fraud_transactions': int(train['isFraud'].sum()),
    'fraud_rate':         round(float(train['isFraud'].mean()), 4),
    'train_date':         'April 19 2026',
    'model_auc':          0.9178,
    'feature_count':      452,
}

analytics = {
    'fraud_by_hour':   fraud_by_hour,
    'vol_by_hour':     vol_by_hour,
    'fraud_by_card4':  fraud_by_card4,
    'fraud_by_domain': fraud_by_domain,
    'fraud_amounts':   fraud_amounts,
    'legit_amounts':   legit_amounts,
    'fraud_ages':      fraud_ages,
    'legit_ages':      legit_ages,
    'shap_importance': shap_importance,
    'summary':         summary,
}

with open(OUT_PATH, 'w') as f:
    json.dump(analytics, f)

print(f"Saved to {OUT_PATH}")
print(f"Total: {summary['total_transactions']:,} transactions")
print(f"Fraud rate: {summary['fraud_rate']*100:.2f}%")
print(f"Top fraud domain: {list(fraud_by_domain.keys())[0]}")