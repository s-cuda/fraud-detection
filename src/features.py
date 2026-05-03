import pandas as pd
import numpy as np


def load_and_merge(txn_path: str, id_path: str) -> pd.DataFrame:
    txn = pd.read_csv(txn_path)
    idf = pd.read_csv(id_path)
    df  = txn.merge(idf, on='TransactionID', how='left')
    print(f"Loaded: {df.shape[0]:,} rows  {df.shape[1]} cols")
    return df


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    missing_pct = df.isnull().mean()
    drop_cols   = missing_pct[missing_pct > threshold].index.tolist()
    drop_cols   = [c for c in drop_cols if c != 'isFraud']
    print(f"Dropping {len(drop_cols)} columns with >{threshold*100:.0f}% missing")
    return df.drop(columns=drop_cols)


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # TransactionDT is a timedelta in seconds from a fixed reference point
    # Extract hour and day so the model can learn time-of-day and day-of-week patterns
    df['hour']        = (df['TransactionDT'] / 3600).astype(int) % 24
    df['day_of_week'] = (df['TransactionDT'] / 86400).astype(int) % 7
    df['is_night']    = df['hour'].between(0, 6).astype(int)
    return df


def build_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    # TransactionAmt is heavily right-skewed — log transform compresses outliers
    df['amt_log'] = np.log1p(df['TransactionAmt'])

    # Raw amount is a weak signal. Amount relative to that card's history is stronger.
    # A $500 charge on a card that usually spends $20 is very different from
    # a $500 charge on a card that regularly spends $1000.
    card_avg = df.groupby('card1')['TransactionAmt'].transform('mean')
    card_std = df.groupby('card1')['TransactionAmt'].transform('std').fillna(1)
    df['amt_to_card_mean_ratio'] = df['TransactionAmt'] / (card_avg + 1)
    df['amt_z_score_card']       = (df['TransactionAmt'] - card_avg) / (card_std + 1)

    return df


def build_card_features(df: pd.DataFrame) -> pd.DataFrame:
    # D1 is days since the card was first used — community reverse-engineered this
    # Cards used within their first month are statistically riskier
    df['card_age']    = df['D1'].fillna(-1)
    df['is_new_card'] = (df['D1'] < 30).astype(int)

    # How many times this card appears in the full dataset
    df['card_txn_count'] = df.groupby('card1')['TransactionID'].transform('count')

    # Frequency encoding replaces card ID with how often it appears in the dataset
    # High frequency = well-established card. Low frequency = rare or newly seen.
    # We use frequency instead of target encoding (fraud rate) because card columns
    # have high cardinality — many cards appear only 3-5 times, making their
    # fraud rate estimate extremely noisy and causing the model to memorize
    # specific cards rather than learning general fraud patterns.
    for col in ['card1', 'card2', 'card3', 'card5']:
        if col in df.columns:
            freq             = df[col].value_counts()
            df[f'{col}_freq'] = df[col].map(freq)

    # card1 + addr1 + D1 gives a more precise customer identifier than card1 alone
    # Two people can share the same card1 value (same issuing bank/BIN)
    # but not the same card1 + billing address + days-since-first-use combination
    df['uid'] = (df['card1'].astype(str) + '_' +
                 df['addr1'].astype(str) + '_' +
                 df['D1'].astype(str))

    uid_amt_mean = df.groupby('uid')['TransactionAmt'].transform('mean')
    uid_amt_std  = df.groupby('uid')['TransactionAmt'].transform('std').fillna(0)

    df['uid_txn_count']   = df.groupby('uid')['TransactionID'].transform('count')
    df['uid_amt_mean']    = uid_amt_mean
    df['uid_amt_std']     = uid_amt_std
    df['amt_to_uid_mean'] = df['TransactionAmt'] / (uid_amt_mean + 1)
    df['uid_amt_z_score'] = (df['TransactionAmt'] - uid_amt_mean) / (uid_amt_std + 1)

    # Multiple addresses or email domains on one card is a known fraud signal
    df['card_unique_addr']  = df.groupby('card1')['addr1'].transform('nunique')
    df['card_unique_email'] = df.groupby('card1')['P_emaildomain'].transform('nunique')
    df['card_amt_std']      = df.groupby('card1')['TransactionAmt'].transform('std').fillna(0)

    # How long this card has been active within the dataset window
    card_first_time             = df.groupby('card1')['TransactionDT'].transform('min')
    df['card_time_since_first'] = df['TransactionDT'] - card_first_time

    df = df.drop(columns=['uid'])
    return df


def build_email_features(df: pd.DataFrame) -> pd.DataFrame:
    df['P_emaildomain'] = df['P_emaildomain'].fillna('unknown')
    df['R_emaildomain'] = df['R_emaildomain'].fillna('unknown')

    # Free email providers are slightly more common in fraud transactions
    free_domains = {'gmail.com', 'yahoo.com', 'hotmail.com',
                    'outlook.com', 'aol.com', 'mail.com'}
    df['P_is_free_email'] = df['P_emaildomain'].isin(free_domains).astype(int)
    df['R_is_free_email'] = df['R_emaildomain'].isin(free_domains).astype(int)

    return df


def encode_device_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'DeviceType' in df.columns:
        # Explicit flags rather than mapping mobile=1/desktop=0
        # which would imply a numeric ordering that does not exist
        df['is_mobile']  = (df['DeviceType'] == 'mobile').astype(float)
        df['is_desktop'] = (df['DeviceType'] == 'desktop').astype(float)
        # Preserve NaN so LightGBM knows value is missing vs not mobile
        missing_mask = df['DeviceType'].isna()
        df.loc[missing_mask, 'is_mobile']  = np.nan
        df.loc[missing_mask, 'is_desktop'] = np.nan
        df = df.drop(columns=['DeviceType'])

    if 'DeviceInfo' in df.columns:
        # DeviceInfo has 1000+ unique strings
        # One-hot would create 1000 sparse columns, most nearly empty
        # Extract OS family which captures the real signal
        df['is_ios']          = df['DeviceInfo'].str.contains('iOS',     na=False).astype(int)
        df['is_windows']      = df['DeviceInfo'].str.contains('Windows', na=False).astype(int)
        df['is_mac']          = df['DeviceInfo'].str.contains('MacOS',   na=False).astype(int)
        df['is_samsung']      = df['DeviceInfo'].str.contains('SAMSUNG', na=False).astype(int)
        df['is_other_device'] = (
            ~df['DeviceInfo'].str.contains('iOS|Windows|MacOS|SAMSUNG', na=True)
        ).astype(int)
        df = df.drop(columns=['DeviceInfo'])

    return df


def encode_identity_features(df: pd.DataFrame) -> pd.DataFrame:
    # id_35 to id_38 are T/F boolean flags from identity verification
    tf_cols = [c for c in ['id_35', 'id_36', 'id_37', 'id_38'] if c in df.columns]
    for col in tf_cols:
        df[col] = df[col].map({'T': 1, 'F': 0})

    # id_34 contains strings like 'match_status:2' — extract the number
    if 'id_34' in df.columns:
        df['id_34'] = df['id_34'].str.extract(r'(-?\d+)').astype(float)

    if 'id_31' in df.columns:
        # id_31 is browser name with version e.g. 'chrome 62.0'
        # Extract browser family — too many unique version strings for one-hot
        df['browser_chrome']  = df['id_31'].str.contains('chrome',  na=False).astype(int)
        df['browser_safari']  = df['id_31'].str.contains('safari',  na=False).astype(int)
        df['browser_firefox'] = df['id_31'].str.contains('firefox', na=False).astype(int)
        df['browser_ie']      = df['id_31'].str.contains('ie',      na=False).astype(int)
        df['browser_samsung'] = df['id_31'].str.contains('samsung', na=False).astype(int)
        df = df.drop(columns=['id_31'])

    if 'id_33' in df.columns:
        # id_33 is screen resolution e.g. '1334x750'
        # Unusual resolutions can indicate automated tools or virtual machines
        df['screen_width']  = df['id_33'].str.extract(r'^(\d+)x').astype(float)
        df['screen_height'] = df['id_33'].str.extract(r'x(\d+)$').astype(float)
        df = df.drop(columns=['id_33'])

    return df


def encode_remaining_strings(df: pd.DataFrame) -> pd.DataFrame:
    # M1-M9 are match verification flags from the payment processor
    m_cols = [c for c in ['M1','M2','M3','M5','M6','M7','M8','M9'] if c in df.columns]
    for col in m_cols:
        df[col] = df[col].map({'T': 1, 'F': 0})

    # id_12, id_16, id_29 — whether the entity was found in a lookup
    found_cols = [c for c in ['id_12', 'id_16', 'id_29'] if c in df.columns]
    for col in found_cols:
        df[col] = df[col].map({'Found': 1, 'NotFound': 0})

    # id_15, id_28 — account or device status
    for col in ['id_15', 'id_28']:
        if col in df.columns:
            df[col] = df[col].map({'Found': 0, 'New': 1, 'Unknown': -1})

    if 'id_30' in df.columns:
        # id_30 is OS with version string e.g. 'Android 7.0', 'Windows 10'
        df['os_android'] = df['id_30'].str.contains('Android', na=False).astype(int)
        df['os_ios']     = df['id_30'].str.contains('iOS',     na=False).astype(int)
        df['os_windows'] = df['id_30'].str.contains('Windows', na=False).astype(int)
        df['os_mac']     = df['id_30'].str.contains('Mac OS',  na=False).astype(int)
        df = df.drop(columns=['id_30'])

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    # Low cardinality categoricals — label encoding works fine here
    # card1/2/3/5 handled above with frequency encoding
    # P_emaildomain/R_emaildomain handled with target encoding in train.py
    cat_cols = [c for c in ['ProductCD', 'card4', 'card6', 'M4'] if c in df.columns]
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Building features...")
    df = drop_high_missing(df)
    df = build_time_features(df)
    df = build_amount_features(df)
    df = build_card_features(df)
    df = build_email_features(df)
    df = encode_device_features(df)
    df = encode_identity_features(df)
    df = encode_remaining_strings(df)
    df = encode_categoricals(df)
    print(f"Feature engineering done. Shape: {df.shape}")
    return df