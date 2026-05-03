import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='shap')

import os
import sys
import time
import json
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(__file__))
from explain import FraudExplainer
from feature_store import FeatureStore

BASE          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(BASE, 'models', 'lgbm_auc0.9178.txt')
ENCODING_PATH = os.path.join(BASE, 'models', 'email_encodings.json')

explainer          = None
feature_store      = None
email_encodings    = {}
model_info         = {}
request_count      = 0
fraud_count        = 0
recent_predictions = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global explainer, feature_store, email_encodings, model_info

    print("Loading model...")
    explainer = FraudExplainer(MODEL_PATH)

    if os.path.exists(ENCODING_PATH):
        with open(ENCODING_PATH) as f:
            email_encodings = json.load(f)
        print(f"Loaded {len(email_encodings)} email domain encodings")
    else:
        print("Warning: email_encodings.json not found, using global fraud rate for email features")

    try:
        feature_store = FeatureStore()
        print("Redis feature store connected.")
    except Exception as e:
        print(f"Warning: Redis not available: {e}")
        feature_store = None

    model_info = {
        'model_path': MODEL_PATH,
        'auc':        0.9178,
        'features':   len(explainer.feature_names),
        'loaded_at':  time.strftime('%Y-%m-%d %H:%M:%S')
    }

    print("API ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title       = "Fraud Detection API",
    description = "Real-time transaction fraud scoring with SHAP explanations",
    version     = "1.0.0",
    lifespan    = lifespan
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")

@app.get("/app")
def serve_dashboard():
    return FileResponse(os.path.join(BASE, "static", "index.html"))


class TransactionRequest(BaseModel):
    TransactionAmt:  float           = Field(..., description="Transaction amount in USD")
    ProductCD:       Optional[str]   = Field(None, description="Product code")
    card4:           Optional[str]   = Field(None, description="Card network e.g. visa, mastercard")
    card6:           Optional[str]   = Field(None, description="Card type e.g. credit, debit")
    P_emaildomain:   Optional[str]   = Field(None, description="Purchaser email domain")
    R_emaildomain:   Optional[str]   = Field(None, description="Recipient email domain")
    TransactionDT:   Optional[float] = Field(None, description="Transaction timedelta in seconds")

    # Engineered features computed during training pipeline.
    # These are filled by the Redis feature store when not provided.
    # If neither the caller nor Redis provides them, LightGBM handles them as missing.
    amt_log:                Optional[float] = None
    amt_to_card_mean_ratio: Optional[float] = None
    amt_z_score_card:       Optional[float] = None
    amt_to_uid_mean:        Optional[float] = None
    uid_amt_z_score:        Optional[float] = None
    card_txn_count:         Optional[float] = None
    card_time_since_first:  Optional[float] = None
    card_unique_email:      Optional[float] = None
    card_unique_addr:       Optional[float] = None
    card_amt_std:           Optional[float] = None
    card_age:               Optional[float] = None
    is_new_card:            Optional[float] = None
    card1_freq:             Optional[float] = None
    card2_freq:             Optional[float] = None
    hour:                   Optional[float] = None
    day_of_week:            Optional[float] = None
    is_night:               Optional[float] = None

    # Display fields used for the dashboard feed.
    # These carry the original human-readable string values before encoding.
    # They are stored in recent_predictions but never fed to the model.
    display_card:   Optional[str] = None
    display_email:  Optional[str] = None
    actual_label:   Optional[int] = None

    class Config:
        extra = 'allow'


class FactorResponse(BaseModel):
    feature:     str
    description: str
    value:       Optional[float]
    shap_value:  float
    direction:   str
    impact:      str


class PredictionResponse(BaseModel):
    fraud_score:  float
    verdict:      str
    top_factors:  List[FactorResponse]
    latency_ms:   float
    request_id:   int


class HealthResponse(BaseModel):
    status:        str
    request_count: int
    fraud_count:   int
    fraud_rate:    float


def prepare_features(request: TransactionRequest) -> pd.DataFrame:
    data = request.model_dump(exclude_none=False)
    df   = pd.DataFrame([data])

    global_mean = 0.035

    # Replace email domain strings with their historical fraud rates from training data.
    # This matches the target encoding applied during the training pipeline.
    for col in ['P_emaildomain', 'R_emaildomain']:
        if col in df.columns:
            domain    = df[col].iloc[0]
            enc_key   = f"{col}_{domain}" if domain else None
            enc_value = email_encodings.get(enc_key, global_mean) if enc_key else global_mean
            df[col]   = enc_value

    # Compute time features from TransactionDT if they were not provided directly.
    if df['hour'].iloc[0] is None and df['TransactionDT'].iloc[0] is not None:
        dt               = float(df['TransactionDT'].iloc[0])
        df['hour']       = int(dt / 3600) % 24
        df['day_of_week'] = int(dt / 86400) % 7
        df['is_night']   = 1 if 0 <= int(dt / 3600) % 24 <= 6 else 0

    if df['amt_log'].iloc[0] is None:
        df['amt_log'] = np.log1p(df['TransactionAmt'].iloc[0])

    # Label encode card and product categoricals to integer codes.
    # This matches the encoding used during feature engineering at training time.
    cat_map = {
        'ProductCD': {'W': 0, 'H': 1, 'C': 2, 'S': 3, 'R': 4},
        'card4':     {'visa': 0, 'mastercard': 1, 'american express': 2, 'discover': 3},
        'card6':     {'debit': 0, 'credit': 1, 'debit or credit': 2, 'charge card': 3},
    }
    for col, mapping in cat_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1)

    # Fill missing features with NaN. LightGBM handles missing values natively
    # by learning which branch to take when a feature is absent.
    for feat in explainer.feature_names:
        if feat not in df.columns:
            df[feat] = np.nan

    # Convert all columns to float. Object dtype breaks SHAP's TreeExplainer.
    return df[explainer.feature_names].apply(pd.to_numeric, errors='coerce')


@app.get("/health", response_model=HealthResponse)
def health():
    fraud_rate = fraud_count / request_count if request_count > 0 else 0.0
    return {
        'status':        'ok',
        'request_count': request_count,
        'fraud_count':   fraud_count,
        'fraud_rate':    round(fraud_rate, 4)
    }


@app.get("/model/info")
def get_model_info():
    return model_info


@app.get("/analytics")
def get_analytics():
    analytics_path = os.path.join(BASE, 'models', 'analytics.json')
    if os.path.exists(analytics_path):
        with open(analytics_path) as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Run src/compute_analytics.py first")


@app.get("/drift/summary")
def get_drift_summary():
    summary_path = os.path.join(BASE, 'models', 'drift_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    return {'status': 'not_run', 'message': 'Run python src/drift_report.py first'}


@app.get("/drift/report")
def get_drift_report():
    from fastapi.responses import HTMLResponse
    report_path = os.path.join(BASE, 'models', 'drift_report.html')
    if os.path.exists(report_path):
        with open(report_path) as f:
            return HTMLResponse(content=f.read())
    raise HTTPException(status_code=404, detail="Run python src/drift_report.py first")


@app.get("/feature-store/stats")
def feature_store_stats():
    if feature_store is None:
        return {'status': 'unavailable'}
    return feature_store.get_stats()


@app.get("/recent")
def get_recent():
    return list(reversed(recent_predictions))


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):
    global request_count, fraud_count

    start = time.time()

    # Enrich the request with card behavioral history from Redis.
    # Redis stores rolling stats per card: average amount, transaction count,
    # unique addresses seen, time since first appearance etc.
    # Only fills in features that the caller did not already provide.
    if feature_store and request.TransactionDT is not None:
        card_id        = str(request.card4 or 'unknown')
        redis_features = feature_store.get_card_features(
            card_id           = card_id,
            current_amount    = request.TransactionAmt,
            current_timestamp = float(request.TransactionDT)
        )
        for feat, value in redis_features.items():
            if getattr(request, feat, None) is None:
                setattr(request, feat, value)

    try:
        X = prepare_features(request)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature preparation failed: {str(e)}")

    try:
        result = explainer.explain(X, top_n=5)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    latency_ms     = round((time.time() - start) * 1000, 2)
    request_count += 1

    if result['fraud_score'] >= 0.7:
        fraud_count += 1

    # Store prediction in the recent list for the dashboard feed.
    # display_card and display_email carry readable values for the UI.
    # actual_label is the ground truth when replaying labeled test data.
    recent_predictions.append({
        'amount':     round(request.TransactionAmt, 2),
        'card':       request.display_card  or str(request.card4  or 'unknown'),
        'email':      request.display_email or str(request.P_emaildomain or 'unknown'),
        'score':      result['fraud_score'],
        'verdict':    result['verdict'],
        'top_factors': result['top_factors'],
        'timestamp':  time.strftime('%H:%M:%S'),
        'latency_ms': latency_ms,
        'actual':     request.actual_label
    })
    if len(recent_predictions) > 20:
        recent_predictions.pop(0)

    # Update Redis with this transaction so future predictions for the same card
    # have an updated behavioral history. Redis failure never blocks a response.
    if feature_store and request.TransactionDT is not None:
        try:
            feature_store.update_card_stats(
                card_id      = str(request.card4 or 'unknown'),
                amount       = request.TransactionAmt,
                timestamp    = float(request.TransactionDT),
                email_domain = str(request.P_emaildomain or ''),
                addr         = ''
            )
        except Exception:
            pass

    return {
        'fraud_score':  result['fraud_score'],
        'verdict':      result['verdict'],
        'top_factors':  result['top_factors'],
        'latency_ms':   latency_ms,
        'request_id':   request_count
    }


@app.post("/predict/batch")
def predict_batch(requests: List[TransactionRequest]):
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Batch size limit is 100")

    results = []
    for req in requests:
        try:
            X      = prepare_features(req)
            result = explainer.explain(X, top_n=3)[0]
            results.append(result)
        except Exception as e:
            results.append({'error': str(e)})

    return results