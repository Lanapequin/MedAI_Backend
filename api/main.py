"""
MedAI REST API — FastAPI backend
POST /predict  → triage classification + SHAP explanation + DB persistence
GET  /predict/history → recent predictions
GET  /health   → service health check
"""

import sys
import time
import unicodedata
import logging
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict

import joblib
import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Project root on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_env import settings  # noqa
from database import get_db, create_tables, SessionLocal  # noqa
import models_db  # noqa
from auth import (  # noqa
    Token, UserOut, UserCreate,
    authenticate_user, create_access_token, create_user,
    get_current_active_user, seed_default_users,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medai")

# ─────────────────────────────────────────────────────────────
# MODEL LOADING (once at startup)
# ─────────────────────────────────────────────────────────────
MODEL_PATH = PROJECT_ROOT / "models" / "triage_model.joblib"

logger.info("Loading MedAI triage model...")
try:
    _pkg = joblib.load(MODEL_PATH)
    _model       = _pkg["model"]
    _preprocessor = _pkg["preprocessor"]
    _feature_names = _pkg["feature_names"]
    _n_classes   = _pkg.get("n_classes", 5)
    _motivo_map  = _pkg.get("motivo_map", {})
    _training_date = _pkg.get("training_date", "unknown")
    _explainer   = shap.TreeExplainer(_model)
    logger.info(f"Model loaded OK — {_n_classes} classes, {len(_feature_names)} features")
except Exception as exc:
    logger.error(f"Failed to load model: {exc}")
    raise

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
NIVEL_INFO: Dict[int, dict] = {
    0: {
        "nombre": "Nivel 1 - Resucitación/Inmediato",
        "descripcion": "Condición que amenaza la vida. Requiere intervención inmediata.",
        "tiempo": "Inmediato (0 minutos)",
        "area": "Área de Shock / Reanimación",
        "color": "#E53E3E",
    },
    1: {
        "nombre": "Nivel 2 - Emergencia/Grave",
        "descripcion": "Condición de alto riesgo. Requiere atención muy rápida.",
        "tiempo": "< 10 minutos",
        "area": "Área de Emergencias",
        "color": "#ED8936",
    },
    2: {
        "nombre": "Nivel 3 - Urgente",
        "descripcion": "Condición que requiere atención pronta pero estable.",
        "tiempo": "< 60 minutos",
        "area": "Consultorios de Urgencias",
        "color": "#ECC94B",
    },
    3: {
        "nombre": "Nivel 4 - Menos Urgente",
        "descripcion": "Condición menor que puede esperar.",
        "tiempo": "< 120 minutos",
        "area": "Consulta Prioritaria",
        "color": "#48BB78",
    },
    4: {
        "nombre": "Nivel 5 - No Urgente",
        "descripcion": "Condición no urgente. Puede esperar o ser atendido en consulta externa.",
        "tiempo": "< 240 minutos",
        "area": "Sala de Espera General / Consulta Externa",
        "color": "#3182CE",
    },
}

MOTIVO_KEYWORDS: Dict[str, int] = {
    "administrativo": 0, "certificado": 0, "receta": 0,
    "covid": 1, "respiratorio": 1, "gripe": 1, "tos": 1, "resfriado": 1,
    "cefalea": 2, "cabeza": 2, "migrana": 2, "jaqueca": 2,
    "dermatologico": 3, "piel": 3, "rash": 3, "alergia": 3, "picazon": 3,
    "disnea": 4, "respirar": 4, "ahogo": 4, "asfixia": 4, "dificultad respiratoria": 4,
    "abdomen": 5, "estomago": 5, "barriga": 5, "abdominal": 5,
    "lumbar": 6, "espalda": 6, "cintura": 6, "columna": 6,
    "torax": 7, "pecho": 7, "toracico": 7, "corazon": 7, "cardiaco": 7, "infarto": 7, "precordial": 7,
    "fiebre": 8, "calentura": 8, "febril": 8,
    "sangre": 9, "hemorragia": 9, "sangrado": 9,
    "herida": 10, "cortada": 10, "raspadura": 10,
    "intoxicacion": 11, "envenenamiento": 11, "toxico": 11,
    "lesion": 12, "golpe leve": 12,
    "mareo": 13, "desmayo": 13, "sincope": 13, "vertigo": 13,
    "embarazo": 14, "parto": 14, "contracciones": 14, "obstetrico": 14,
    "dengue": 16,
    "malaria": 17, "paludismo": 17,
    "trauma": 18, "golpe": 18, "caida": 18, "accidente": 18, "fractura": 18,
    "vomito": 19, "diarrea": 19, "nausea": 19,
    "orina": 5, "urinario": 5, "rinon": 5, "vejiga": 5,
}


def _normalize(text: str) -> str:
    text = text.lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _encode_motivo(text: str) -> int:
    norm = _normalize(text)
    for kw, code in MOTIVO_KEYWORDS.items():
        if kw in norm:
            return code
    return 15  # Otros


# ─────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────────
class SignosVitales(BaseModel):
    frecuencia_cardiaca: int = Field(..., ge=20, le=250)
    frecuencia_respiratoria: int = Field(..., ge=4, le=70)
    temperatura: float = Field(..., ge=32.0, le=44.0)
    spO2: int = Field(..., ge=40, le=100)
    presion_sistolica: int = Field(..., ge=40, le=300)
    presion_diastolica: int = Field(..., ge=20, le=200)
    dolor: int = Field(..., ge=0, le=10)


class Comorbilidades(BaseModel):
    hta: bool = False
    dm2: bool = False
    epoc: bool = False
    irc: bool = False
    cardiopatia: bool = False
    obesidad: bool = False
    cancer: bool = False
    embarazo: bool = False


class PredictRequest(BaseModel):
    edad: int = Field(..., ge=0, le=120)
    sexo: str = Field(..., pattern="^[MF]$")
    modo_llegada: str
    motivo_consulta: str = Field(..., min_length=2, max_length=200)
    signos_vitales: SignosVitales
    comorbilidades: Comorbilidades = Comorbilidades()
    requiere_labs: bool = False
    requiere_imagenes: bool = False
    es_prioritario: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "edad": 65,
                "sexo": "M",
                "modo_llegada": "Ambulancia",
                "motivo_consulta": "Dolor en el pecho desde hace 2 horas",
                "signos_vitales": {
                    "frecuencia_cardiaca": 110,
                    "frecuencia_respiratoria": 24,
                    "temperatura": 37.2,
                    "spO2": 88,
                    "presion_sistolica": 90,
                    "presion_diastolica": 60,
                    "dolor": 8,
                },
                "comorbilidades": {"hta": True, "cardiopatia": True},
                "requiere_labs": True,
                "requiere_imagenes": True,
            }
        }


class FactorRiesgo(BaseModel):
    factor: str
    descripcion: str
    severidad: str


class SHAPFeature(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str  # "aumenta" | "disminuye"


class PredictResponse(BaseModel):
    # MTS classification
    nivel_triage: str
    nivel_codigo: int
    descripcion: str
    confianza: float
    probabilidades: Dict[str, float]
    # Timing
    tiempo_atencion_recomendado: str
    area_recomendada: str
    # SHAP explainability
    shap_base_value: float
    shap_features: List[SHAPFeature]
    shap_global_importance: Dict[str, float]
    # Clinical risk flags (rule-based)
    factores_riesgo: List[FactorRiesgo]
    # Meta
    timestamp: str
    prediction_id: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    timestamp: str


class PredictionRecord(BaseModel):
    prediction_id: int
    patient_id: int
    nivel_codigo: int
    nivel_triage: str
    confianza: float
    timestamp: str
    edad: int
    sexo: str
    motivo_consulta: str


# ─────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="MedAI Triage API",
    description=(
        "Explainable AI-Based Clinical Triage Support System.\n\n"
        "**POST /predict** — full triage classification with SHAP explanation.\n\n"
        "**GET /predict/history** — recent predictions from the database."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    create_tables()
    with SessionLocal() as db:
        seed_default_users(db)
    logger.info("Database tables created/verified. Default users seeded.")


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def _build_feature_df(req: PredictRequest) -> pd.DataFrame:
    sv = req.signos_vitales
    co = req.comorbilidades
    return pd.DataFrame([{
        "edad":                   req.edad,
        "frecuencia_cardiaca":    sv.frecuencia_cardiaca,
        "frecuencia_respiratoria":sv.frecuencia_respiratoria,
        "temperatura":            sv.temperatura,
        "spO2":                   sv.spO2,
        "presion_sistolica":      sv.presion_sistolica,
        "presion_diastolica":     sv.presion_diastolica,
        "dolor":                  sv.dolor,
        "sexo_encoded":           1 if req.sexo == "M" else 0,
        "motivo_encoded":         _encode_motivo(req.motivo_consulta),
        "diabetes":               int(co.dm2),
        "hipertension":           int(co.hta),
        "enfermedad_cardiaca":    int(co.cardiopatia),
        "enfermedad_respiratoria":int(co.epoc),
        "inmunosupresion":        int(co.cancer or co.irc),
    }])


def _compute_shap(X_proc: np.ndarray, pred_class: int) -> dict:
    shap_vals = _explainer.shap_values(X_proc)          # (1, 15, 5)
    base_vals = np.array(_explainer.expected_value)     # (5,)

    shap_arr = np.array(shap_vals)                      # (1, 15, 5)
    base_value = float(base_vals[pred_class])

    # Per-feature SHAP for predicted class
    class_shap = shap_arr[0, :, pred_class]             # (15,)

    features = []
    for i, fname in enumerate(_feature_names):
        sv = float(class_shap[i])
        features.append({
            "feature":    fname,
            "value":      float(X_proc[0, i]),
            "shap_value": sv,
            "direction":  "aumenta" if sv > 0 else "disminuye",
        })

    # Sort by absolute SHAP value descending
    features.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    # Global importance: mean |SHAP| across all classes
    global_importance = {
        fname: float(np.abs(shap_arr[0, i, :]).mean())
        for i, fname in enumerate(_feature_names)
    }

    return {
        "base_value":         base_value,
        "features":           features,
        "global_importance":  global_importance,
        "shap_matrix":        shap_arr[0].tolist(),   # 15×5 for storage
    }


def _identify_risk_factors(req: PredictRequest) -> List[FactorRiesgo]:
    factors = []
    sv = req.signos_vitales

    if sv.spO2 < 90:
        factors.append(FactorRiesgo(factor="Hipoxia severa", descripcion=f"SpO₂ = {sv.spO2}%", severidad="alta"))
    elif sv.spO2 < 94:
        factors.append(FactorRiesgo(factor="Hipoxia leve", descripcion=f"SpO₂ = {sv.spO2}%", severidad="media"))

    if sv.presion_sistolica < 90:
        factors.append(FactorRiesgo(factor="Hipotensión", descripcion=f"PA = {sv.presion_sistolica}/{sv.presion_diastolica} mmHg", severidad="alta"))
    elif sv.presion_sistolica > 180:
        factors.append(FactorRiesgo(factor="Crisis hipertensiva", descripcion=f"PA = {sv.presion_sistolica}/{sv.presion_diastolica} mmHg", severidad="alta"))

    if sv.frecuencia_cardiaca > 120:
        factors.append(FactorRiesgo(factor="Taquicardia severa", descripcion=f"FC = {sv.frecuencia_cardiaca} lpm", severidad="alta"))
    elif sv.frecuencia_cardiaca < 50:
        factors.append(FactorRiesgo(factor="Bradicardia severa", descripcion=f"FC = {sv.frecuencia_cardiaca} lpm", severidad="alta"))

    if sv.temperatura > 39.5:
        factors.append(FactorRiesgo(factor="Fiebre alta", descripcion=f"Temperatura = {sv.temperatura}°C", severidad="alta"))
    elif sv.temperatura < 35.0:
        factors.append(FactorRiesgo(factor="Hipotermia", descripcion=f"Temperatura = {sv.temperatura}°C", severidad="alta"))

    if sv.frecuencia_respiratoria > 24:
        factors.append(FactorRiesgo(factor="Taquipnea", descripcion=f"FR = {sv.frecuencia_respiratoria} rpm", severidad="media"))

    if sv.dolor >= 8:
        factors.append(FactorRiesgo(factor="Dolor severo", descripcion=f"EVA = {sv.dolor}/10", severidad="alta"))

    co = req.comorbilidades
    n_co = sum([co.hta, co.dm2, co.epoc, co.irc, co.cardiopatia, co.cancer])
    if req.edad >= 65 and n_co >= 2:
        factors.append(FactorRiesgo(factor="Paciente geriátrico multimórbido", descripcion=f"Edad {req.edad} años, {n_co} comorbilidades", severidad="media"))

    if req.modo_llegada == "Ambulancia":
        factors.append(FactorRiesgo(factor="Llegada en ambulancia", descripcion="Traslado por servicio de emergencia", severidad="media"))

    return factors


def _persist(req: PredictRequest, pred_class: int, confianza: float,
             probabilidades: dict, shap_data: dict, latency_ms: float,
             db: Session) -> int:
    try:
        sv = req.signos_vitales
        co = req.comorbilidades
        info = NIVEL_INFO[pred_class]

        patient = models_db.Patient(
            edad=req.edad, sexo=req.sexo,
            modo_llegada=req.modo_llegada,
            motivo_consulta=req.motivo_consulta,
            frecuencia_cardiaca=sv.frecuencia_cardiaca,
            frecuencia_respiratoria=sv.frecuencia_respiratoria,
            temperatura=sv.temperatura, spO2=sv.spO2,
            presion_sistolica=sv.presion_sistolica,
            presion_diastolica=sv.presion_diastolica,
            dolor=sv.dolor,
            hta=co.hta, dm2=co.dm2, epoc=co.epoc, irc=co.irc,
            cardiopatia=co.cardiopatia, obesidad=co.obesidad,
            cancer=co.cancer, embarazo=co.embarazo,
            requiere_labs=req.requiere_labs,
            requiere_imagenes=req.requiere_imagenes,
            es_prioritario=req.es_prioritario,
        )
        db.add(patient)
        db.flush()

        prediction = models_db.Prediction(
            patient_id=patient.id,
            nivel_triage=info["nombre"],
            nivel_codigo=pred_class + 1,
            probabilidades=probabilidades,
            confianza=confianza,
            descripcion=info["descripcion"],
            tiempo_atencion_recomendado=info["tiempo"],
            area_recomendada=info["area"],
            inference_latency_ms=latency_ms,
        )
        db.add(prediction)
        db.flush()

        explanation = models_db.Explanation(
            prediction_id=prediction.id,
            base_value=shap_data["base_value"],
            shap_values={f["feature"]: f["shap_value"] for f in shap_data["features"]},
            shap_values_all_classes=shap_data["shap_matrix"],
            top_features=shap_data["features"][:10],
            predicted_class_idx=pred_class,
        )
        db.add(explanation)
        db.commit()
        return prediction.id
    except Exception as exc:
        db.rollback()
        logger.warning(f"DB persistence failed: {exc}")
        return -1


# ─────────────────────────────────────────────────────────────
# ENDPOINTS — Auth
# ─────────────────────────────────────────────────────────────
@app.post("/auth/login", response_model=Token, tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Obtener token JWT con usuario y contraseña."""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": user.username, "role": user.role})
    return Token(access_token=token, token_type="bearer")


@app.post("/auth/register", response_model=UserOut, tags=["Auth"])
def register(data: UserCreate, db: Session = Depends(get_db)):
    """Registrar una nueva cuenta. Abierto — no requiere token."""
    from auth import get_user_by_username
    if get_user_by_username(db, data.username):
        raise HTTPException(status_code=400, detail="El nombre de usuario ya existe")
    return create_user(db, data)


@app.get("/auth/me", response_model=UserOut, tags=["Auth"])
def me(current_user: models_db.User = Depends(get_current_active_user)):
    """Información del usuario autenticado."""
    return current_user


# ─────────────────────────────────────────────────────────────
# ENDPOINTS — Health (público)
# ─────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        version="2.0.0",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Triage"])
def predict(
    req: PredictRequest,
    db: Session = Depends(get_db),
    current_user: models_db.User = Depends(get_current_active_user),
):
    """
    Full MTS triage classification with SHAP explanation.

    Returns urgency level (1–5), class probability distribution,
    per-feature SHAP values for the predicted class, and rule-based
    clinical risk flags. Result is persisted to PostgreSQL.
    """
    t0 = time.perf_counter()

    # 1. Preprocess
    X_df = _build_feature_df(req)
    try:
        X_proc = _preprocessor.transform(X_df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {exc}")

    # 2. Inference
    proba_raw = _model.predict_proba(X_proc)[0]   # (5,)
    pred_class = int(np.argmax(proba_raw))

    # 3. Calibrate probabilities (temperature scaling for realistic confidence)
    temp = 2.0
    log_p = np.log(np.clip(proba_raw, 1e-10, 1 - 1e-10))
    exp_scaled = np.exp(log_p / temp - np.max(log_p / temp))
    proba = exp_scaled / exp_scaled.sum()

    confianza = round(float(proba[pred_class]) * 100, 1)
    probabilidades = {f"Nivel {i+1}": round(float(p) * 100, 1) for i, p in enumerate(proba)}

    # 4. SHAP explanation (synchronous — TreeExplainer is fast enough)
    try:
        shap_data = _compute_shap(X_proc, pred_class)
    except Exception as exc:
        logger.warning(f"SHAP failed: {exc}")
        shap_data = {
            "base_value": 0.0,
            "features": [{"feature": f, "value": 0.0, "shap_value": 0.0, "direction": "disminuye"}
                         for f in _feature_names],
            "global_importance": {f: 0.0 for f in _feature_names},
            "shap_matrix": [[0.0] * 5] * len(_feature_names),
        }

    # 5. Rule-based risk factors
    factores_riesgo = _identify_risk_factors(req)

    latency_ms = (time.perf_counter() - t0) * 1000

    # 6. Persist to DB
    pred_id = _persist(req, pred_class, confianza, probabilidades, shap_data, latency_ms, db)

    info = NIVEL_INFO[pred_class]
    return PredictResponse(
        nivel_triage=info["nombre"],
        nivel_codigo=pred_class + 1,
        descripcion=info["descripcion"],
        confianza=confianza,
        probabilidades=probabilidades,
        tiempo_atencion_recomendado=info["tiempo"],
        area_recomendada=info["area"],
        shap_base_value=shap_data["base_value"],
        shap_features=[SHAPFeature(**f) for f in shap_data["features"]],
        shap_global_importance=shap_data["global_importance"],
        factores_riesgo=factores_riesgo,
        timestamp=datetime.utcnow().isoformat(),
        prediction_id=pred_id if pred_id > 0 else None,
    )


@app.post("/clasificar", response_model=PredictResponse, tags=["Triage"])
def clasificar(
    req: PredictRequest,
    db: Session = Depends(get_db),
    current_user: models_db.User = Depends(get_current_active_user),
):
    """Alias of POST /predict for backward compatibility."""
    return predict(req, db, current_user)


@app.get("/predict/history", response_model=List[PredictionRecord], tags=["Triage"])
def history(
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: models_db.User = Depends(get_current_active_user),
):
    """Return the most recent predictions from the database."""
    rows = (
        db.query(models_db.Prediction, models_db.Patient)
        .join(models_db.Patient, models_db.Prediction.patient_id == models_db.Patient.id)
        .order_by(models_db.Prediction.created_at.desc())
        .limit(min(limit, 100))
        .all()
    )
    return [
        PredictionRecord(
            prediction_id=pred.id,
            patient_id=pat.id,
            nivel_codigo=pred.nivel_codigo,
            nivel_triage=pred.nivel_triage or "",
            confianza=pred.confianza or 0.0,
            timestamp=pred.created_at.isoformat(),
            edad=pat.edad,
            sexo=pat.sexo,
            motivo_consulta=pat.motivo_consulta,
        )
        for pred, pat in rows
    ]


@app.get("/info/model", tags=["Info"])
def model_info(current_user: models_db.User = Depends(get_current_active_user)):
    return {
        "model_type": "XGBoost",
        "n_classes": _n_classes,
        "n_features": len(_feature_names),
        "features": _feature_names,
        "training_date": _training_date[:10],
        "dataset": "triage_choco_100k_balanceado.csv",
        "mts_levels": {str(k + 1): v["nombre"] for k, v in NIVEL_INFO.items()},
    }


@app.get("/info/niveles", tags=["Info"])
def niveles_info(current_user: models_db.User = Depends(get_current_active_user)):
    return NIVEL_INFO


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.backend_host,
                port=settings.backend_port, reload=False)
