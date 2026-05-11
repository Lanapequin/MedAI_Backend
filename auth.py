"""
auth.py
JWT authentication utilities for MedAI API.
"""
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config_env import settings
from database import get_db
import models_db

# ── Crypto setup ──────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ── Pydantic schemas ──────────────────────────────────────────────
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool

    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    password: str
    role: str = "nurse"


# ── Password helpers ──────────────────────────────────────────────
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── JWT helpers ───────────────────────────────────────────────────
def create_access_token(data: dict) -> str:
    payload = data.copy()
    expire = datetime.utcnow() + timedelta(
        minutes=settings.jwt_access_token_expire_minutes
    )
    payload.update({"exp": expire})
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


# ── DB helpers ────────────────────────────────────────────────────
def get_user_by_username(db: Session, username: str) -> Optional[models_db.User]:
    return db.query(models_db.User).filter(models_db.User.username == username).first()


def authenticate_user(db: Session, username: str, password: str) -> Optional[models_db.User]:
    user = get_user_by_username(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def create_user(db: Session, data: UserCreate) -> models_db.User:
    user = models_db.User(
        username=data.username,
        email=data.email,
        full_name=data.full_name,
        hashed_password=hash_password(data.password),
        role=data.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ── FastAPI dependency ────────────────────────────────────────────
def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> models_db.User:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido o expirado",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_error
    except JWTError:
        raise credentials_error

    user = get_user_by_username(db, username)
    if user is None or not user.is_active:
        raise credentials_error
    return user


def get_current_active_user(current_user: models_db.User = Depends(get_current_user)):
    return current_user


# ── Seed default users ────────────────────────────────────────────
DEFAULT_USERS = [
    {"username": "admin",    "email": "admin@medai.local",    "full_name": "Administrador",       "password": "admin123",    "role": "admin"},
    {"username": "nurse1",   "email": "nurse1@medai.local",   "full_name": "Enfermera de Triage", "password": "nurse123",    "role": "nurse"},
    {"username": "doctor1",  "email": "doctor1@medai.local",  "full_name": "Médico Urgencias",    "password": "doctor123",   "role": "physician"},
]


def seed_default_users(db: Session):
    for data in DEFAULT_USERS:
        if not get_user_by_username(db, data["username"]):
            create_user(db, UserCreate(**data))
