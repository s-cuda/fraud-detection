import shap
import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='shap')

# Feature name mappings — translates raw column names into readable descriptions
# Based on community reverse engineering of the IEEE-CIS dataset
FEATURE_DESCRIPTIONS = {
    'C13':                    'number of billing addresses on this payment method',
    'C14':                    'payment method address count (variant)',
    'C1':                     'count of addresses associated with card',
    'C11':                    'transaction count feature',
    'C5':                     'count-based risk feature',
    'C6':                     'count-based risk feature (variant)',
    'V70':                    'Vesta risk score (device/identity group)',
    'V294':                   'Vesta risk score (behavioral group)',
    'V258':                   'Vesta risk score (transaction group)',
    'V308':                   'Vesta risk score (payment group)',
    'TransactionAmt':         'transaction amount',
    'amt_log':                'transaction amount (log scaled)',
    'amt_to_card_mean_ratio': 'ratio of this amount to card average amount',
    'amt_z_score_card':       'how many std deviations this amount is from card average',
    'amt_to_uid_mean':        'ratio of this amount to customer average amount',
    'uid_amt_z_score':        'how many std deviations from customer average amount',
    'card_txn_count':         'total transactions seen for this card',
    'card_time_since_first':  'seconds since this card was first seen',
    'card_unique_email':      'number of different email domains used with this card',
    'card_unique_addr':       'number of different addresses used with this card',
    'card_amt_std':           'standard deviation of amounts for this card',
    'card_age':               'days since card was first used (D1)',
    'is_new_card':            'card is less than 30 days old',
    'card1_freq':             'how commonly this card type appears in the dataset',
    'card2_freq':             'how commonly this card variant appears',
    'D1':                     'days since card was first used',
    'D2':                     'days since last transaction on this card',
    'D4':                     'days between transactions (variant)',
    'D10':                    'days-based feature',
    'D15':                    'days-based feature (variant)',
    'P_emaildomain':          'purchaser email domain fraud rate',
    'R_emaildomain':          'recipient email domain fraud rate',
    'P_is_free_email':        'purchaser uses free email provider',
    'R_is_free_email':        'recipient uses free email provider',
    'M4':                     'transaction match flag',
    'M5':                     'billing match flag',
    'M6':                     'identity match flag',
    'hour':                   'hour of day transaction occurred',
    'day_of_week':            'day of week transaction occurred',
    'is_night':               'transaction occurred between midnight and 6am',
    'card6':                  'card type (credit/debit)',
    'card4':                  'card network (visa/mastercard)',
    'is_mobile':              'transaction made on mobile device',
    'is_desktop':             'transaction made on desktop',
    'browser_chrome':         'transaction made using Chrome browser',
    'browser_safari':         'transaction made using Safari browser',
    'os_windows':             'device runs Windows',
    'os_android':             'device runs Android',
    'os_ios':                 'device runs iOS',
    'screen_width':           'device screen width',
    'uid_txn_count':          'total transactions for this customer identifier',
    'uid_amt_mean':           'average transaction amount for this customer',
}


def get_feature_description(feature_name: str) -> str:
    return FEATURE_DESCRIPTIONS.get(feature_name, feature_name.replace('_', ' '))


class FraudExplainer:
    """
    Wraps a trained LightGBM model and a SHAP TreeExplainer.
    Provides predict() and explain() methods for the API layer.
    """

    def __init__(self, model_path: str):
        print(f"Loading model from {model_path}")
        self.model    = lgb.Booster(model_file=model_path)
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_names = self.model.feature_name()
        print(f"Model loaded. Features: {len(self.feature_names)}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def explain(self, X: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """
        For each row in X, return the top_n features that most influenced
        the prediction, with their direction and contribution magnitude.

        Returns a list of dicts, one per row:
        {
            'fraud_score': float,
            'verdict': str,
            'top_factors': [
                {
                    'feature': str,
                    'description': str,
                    'value': float,
                    'shap_value': float,
                    'direction': str,   # 'increases_risk' or 'decreases_risk'
                    'impact': str       # 'high' / 'medium' / 'low'
                },
                ...
            ]
        }
        """
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        fraud_scores = self.predict(X)
        results      = []

        for i in range(len(X)):
            score      = float(fraud_scores[i])
            shap_row   = shap_values[i]
            feature_row = X.iloc[i]

            # Sort features by absolute SHAP value
            indices = np.argsort(np.abs(shap_row))[::-1][:top_n]

            factors = []
            for idx in indices:
                sv          = float(shap_row[idx])
                feat_name   = self.feature_names[idx]
                feat_val    = float(feature_row.iloc[idx]) if not pd.isna(feature_row.iloc[idx]) else None
                abs_sv      = abs(sv)

                factors.append({
                    'feature':     feat_name,
                    'description': get_feature_description(feat_name),
                    'value':       feat_val,
                    'shap_value':  round(sv, 4),
                    'direction':   'increases_risk' if sv > 0 else 'decreases_risk',
                    'impact':      'high' if abs_sv > 0.1 else 'medium' if abs_sv > 0.05 else 'low'
                })

            results.append({
                'fraud_score': round(score, 4),
                'verdict':     _verdict(score),
                'top_factors': factors
            })

        return results

    def explain_text(self, X: pd.DataFrame, top_n: int = 5) -> List[str]:
        """
        Plain English explanation for each row.
        Used for the Gradio demo and API text responses.
        """
        explanations = self.explain(X, top_n=top_n)
        texts        = []

        for exp in explanations:
            score   = exp['fraud_score']
            verdict = exp['verdict']
            lines   = [f"Fraud score: {score:.3f} — {verdict}", ""]

            risk_up   = [f for f in exp['top_factors'] if f['direction'] == 'increases_risk']
            risk_down = [f for f in exp['top_factors'] if f['direction'] == 'decreases_risk']

            if risk_up:
                lines.append("Risk factors (pushing score UP):")
                for f in risk_up:
                    val_str = f"{f['value']:.2f}" if f['value'] is not None else "missing"
                    lines.append(f"  • {f['description']} = {val_str}  (+{f['shap_value']:.3f})")

            if risk_down:
                lines.append("\nProtective factors (pushing score DOWN):")
                for f in risk_down:
                    val_str = f"{f['value']:.2f}" if f['value'] is not None else "missing"
                    lines.append(f"  • {f['description']} = {val_str}  ({f['shap_value']:.3f})")

            texts.append("\n".join(lines))

        return texts


def _verdict(score: float) -> str:
    if score >= 0.7:
        return "HIGH RISK — recommend block"
    elif score >= 0.4:
        return "MEDIUM RISK — recommend review"
    elif score >= 0.2:
        return "LOW RISK — monitor"
    else:
        return "VERY LOW RISK — likely legitimate"


if __name__ == "__main__":
    # Quick test — load model and explain a few test transactions
    import os
    import sys
    sys.path.append(os.path.dirname(__file__))
    from features import load_and_merge, build_all_features

    BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE, 'models', 'lgbm_auc0.9178.txt')
    TXN_PATH   = os.path.join(BASE, 'data', 'train_transaction.csv')
    ID_PATH    = os.path.join(BASE, 'data', 'train_identity.csv')

    # Load data
    df = load_and_merge(TXN_PATH, ID_PATH)
    df = build_all_features(df)
    df = df.sort_values('TransactionDT')

    # Reproduce split and encoding
    split_idx = int(len(df) * 0.8)
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()

    for col in ['P_emaildomain', 'R_emaildomain']:
        fraud_rate    = train_df.groupby(col)['isFraud'].mean()
        global_mean   = train_df['isFraud'].mean()
        test_df[col]  = test_df[col].map(fraud_rate).fillna(global_mean)
        train_df[col] = train_df[col].map(fraud_rate).fillna(global_mean)

    drop_cols = ['TransactionID', 'isFraud', 'TransactionDT',
                 'card1', 'card2', 'card3', 'card5']
    drop_cols = [c for c in drop_cols if c in test_df.columns]
    X_test    = test_df.drop(columns=drop_cols)
    y_test    = test_df['isFraud']

    # Load explainer
    explainer = FraudExplainer(MODEL_PATH)

    # Find one fraud and one legit transaction
    fraud_idx = y_test[y_test == 1].index[0]
    legit_idx = y_test[y_test == 0].index[0]

    fraud_row = X_test.loc[[fraud_idx]]
    legit_row = X_test.loc[[legit_idx]]

    print("\n" + "="*60)
    print("FRAUD TRANSACTION EXPLANATION")
    print("="*60)
    print(explainer.explain_text(fraud_row)[0])

    print("\n" + "="*60)
    print("LEGIT TRANSACTION EXPLANATION")
    print("="*60)
    print(explainer.explain_text(legit_row)[0])