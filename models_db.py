"""
models_db.py
SQLAlchemy ORM models for MedAI persistence layer.
Entities: User, Patient, Prediction, Explanation.
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, Text, JSON
)
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(200), unique=True, nullable=False)
    full_name = Column(String(200))
    hashed_password = Column(String(200), nullable=False)
    role = Column(String(20), default="nurse")  # nurse | physician | admin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    # Demographics
    edad = Column(Integer, nullable=False)
    sexo = Column(String(1), nullable=False)
    modo_llegada = Column(String(50))
    motivo_consulta = Column(Text, nullable=False)
    # Vital signs (stored as individual columns for queryability)
    frecuencia_cardiaca = Column(Integer)
    frecuencia_respiratoria = Column(Integer)
    temperatura = Column(Float)
    spO2 = Column(Integer)
    presion_sistolica = Column(Integer)
    presion_diastolica = Column(Integer)
    dolor = Column(Integer)
    # Comorbidities
    hta = Column(Boolean, default=False)
    dm2 = Column(Boolean, default=False)
    epoc = Column(Boolean, default=False)
    irc = Column(Boolean, default=False)
    cardiopatia = Column(Boolean, default=False)
    obesidad = Column(Boolean, default=False)
    cancer = Column(Boolean, default=False)
    embarazo = Column(Boolean, default=False)
    # Extra flags
    requiere_labs = Column(Boolean, default=False)
    requiere_imagenes = Column(Boolean, default=False)
    es_prioritario = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    prediction = relationship("Prediction", back_populates="patient", uselist=False)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    # MTS classification
    nivel_triage = Column(String(100))
    nivel_codigo = Column(Integer, nullable=False)
    # Probabilities over 5 MTS levels (JSON: {"Nivel 1": 12.3, ...})
    probabilidades = Column(JSON)
    confianza = Column(Float)
    descripcion = Column(Text)
    tiempo_atencion_recomendado = Column(String(100))
    area_recomendada = Column(String(200))
    # Performance tracking
    inference_latency_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="prediction")
    explanation = relationship("Explanation", back_populates="prediction", uselist=False)


class Explanation(Base):
    __tablename__ = "explanations"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    # SHAP payload (JSONB-equivalent via JSON column)
    base_value = Column(Float)                  # φ₀ — expected model output
    shap_values = Column(JSON)                  # {feature: shap_value} for predicted class
    shap_values_all_classes = Column(JSON)      # full 15×5 matrix
    top_features = Column(JSON)                 # ranked list [{feature, value, shap, direction}]
    predicted_class_idx = Column(Integer)       # 0-indexed
    created_at = Column(DateTime, default=datetime.utcnow)

    prediction = relationship("Prediction", back_populates="explanation")
