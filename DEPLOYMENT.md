# MedAI — Deployment & Usage Guide

## Architecture Overview

```
Browser → EC2 (Nginx + FastAPI + XGBoost + SHAP + PostgreSQL)
Browser → CloudFront/S3 or Amplify (React frontend)
```

The system follows the C4 Container Diagram from the paper. In this student deployment,
PostgreSQL runs on the same EC2 instance instead of RDS — the logical architecture
(4 containers: Frontend, API, ML Model, Database) remains identical. Only the physical
co-location changes.

> **Note on paper diagrams:** The C4 Context and Container diagrams do not need to change.
> The Deployment Architecture diagram in the paper shows RDS as a separate service, which
> reflects the production target. This guide deploys the database on EC2 for simplicity
> under student account constraints.

---

## Local Development

### Prerequisites
- macOS with Homebrew (or Linux)
- PostgreSQL 14+
- Node.js 18+
- Python 3.11

### 1. PostgreSQL Setup

```bash
# macOS
brew install postgresql@18
brew services start postgresql@18
psql -h localhost -U $USER -d postgres -c "CREATE DATABASE medai;"

# Linux (Ubuntu/Debian)
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo -u postgres psql -c "CREATE USER medai WITH PASSWORD 'medaipass';"
sudo -u postgres psql -c "CREATE DATABASE medai OWNER medai;"
```

### 2. Backend Setup

```bash
cd MedAI_Backend

# macOS — instalar Python 3.11 y dependencias de sistema
brew install python@3.11 libomp expat

# Crear virtualenv (macOS: requiere DYLD prefix por conflicto de libexpat)
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib \
  /opt/homebrew/bin/python3.11 -m venv venv

# Instalar paquetes
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib \
  venv/bin/pip install --upgrade pip
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib \
  venv/bin/pip install -r requirements.txt

# Configurar entorno
cp .env.example .env
# Editar .env — ajusta DATABASE_URL a tu usuario de sistema:
#   DATABASE_URL=postgresql://TU_USUARIO@localhost/medai   (macOS sin contraseña)
#   DATABASE_URL=postgresql://medai:medaipass@localhost/medai  (Linux con contraseña)

# Entrenar el modelo (~5 minutos, solo la primera vez)
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib:/opt/homebrew/opt/libomp/lib \
  venv/bin/python train_model.py

# Iniciar el backend
./start_dev.sh
```

> **Nota macOS**: `start_dev.sh` aplica el `DYLD_LIBRARY_PATH` automáticamente.
> En Linux no es necesario ningún prefijo especial.

Backend: http://localhost:8000 | Docs: http://localhost:8000/docs

### 3. Frontend Setup

```bash
cd MedAI_Frontend
npm install
cp .env.example .env.local   # ya contiene NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

Frontend: http://localhost:3000

---

## Testing Guide

### API Endpoints (con JWT)

```bash
# 1. Registrar usuario
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"test","email":"test@medai.local","password":"pass123","role":"nurse"}'

# 2. Login — obtener token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test&password=pass123" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 3. Predicción — paciente crítico (esperado Nivel 1)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "edad": 72, "sexo": "F", "modo_llegada": "Ambulancia",
    "motivo_consulta": "Disnea severa y SpO2 baja",
    "signos_vitales": {
      "frecuencia_cardiaca": 135, "frecuencia_respiratoria": 32,
      "temperatura": 38.8, "spO2": 82,
      "presion_sistolica": 80, "presion_diastolica": 50, "dolor": 7
    },
    "comorbilidades": {"epoc": true, "cardiopatia": true, "hta": true}
  }'

# 4. Historial
curl http://localhost:8000/predict/history -H "Authorization: Bearer $TOKEN"
```

### Verificar base de datos

```bash
psql -h localhost -U $USER -d medai -c "
  SELECT pr.id, pat.edad, pat.motivo_consulta,
         pr.nivel_codigo, pr.confianza, e.base_value
  FROM patients pat
  JOIN predictions pr ON pr.patient_id = pat.id
  JOIN explanations e  ON e.prediction_id = pr.id
  ORDER BY pr.id DESC LIMIT 5;"
```

### Frontend Checklist

1. Ir a http://localhost:3000 → redirige a `/login`
2. Crear cuenta en `/register`
3. Completar formulario de triage con todos los campos (incluido nivel de dolor)
4. Clic en **"Realizar Clasificación de Triage"**
5. Verificar resultado: nivel, distribución de probabilidades, barras SHAP
6. `/pacientes` → historial de clasificaciones
7. `/informes` → estadísticas del turno

---

## AWS Deployment (Cuenta de Estudiante)

### Arquitectura simplificada para cuenta de estudiante

```
Browser
  │
  ├──→ AWS Amplify (React frontend — gratuito en Free Tier)
  │
  └──→ EC2 t2.micro o t3.small
            ├── Nginx (reverse proxy HTTPS)
            ├── FastAPI + XGBoost + SHAP (puerto 8000, local)
            └── PostgreSQL (puerto 5432, local, solo escucha localhost)
```

PostgreSQL corre en el mismo EC2. Esto es completamente válido para un prototipo
académico y no requiere RDS (que no está disponible en muchas cuentas de estudiante).

---

### Paso 1: Crear la instancia EC2

En la consola de AWS:

1. **EC2 → Launch Instance**
2. **Nombre**: `medai-server`
3. **AMI**: Ubuntu Server 24.04 LTS (Free Tier eligible)
4. **Instance type**: `t2.micro` (free tier) o `t3.small` (recomendado para XGBoost)
5. **Key pair**: crear uno nuevo → descargar el `.pem`
6. **Security Group** — agregar estas reglas de entrada:

   | Tipo | Puerto | Origen | Para qué |
   |------|--------|--------|----------|
   | SSH | 22 | Tu IP | Conexión SSH |
   | HTTP | 80 | 0.0.0.0/0 | Redirect a HTTPS |
   | HTTPS | 443 | 0.0.0.0/0 | API pública |

7. **Storage**: 20 GB (el modelo pesa ~12 MB, el CSV ~13 MB)
8. Clic **Launch Instance**

> **Sobre el tipo de instancia**: `t2.micro` tiene 1 GB de RAM, lo que es justo para
> cargar el modelo XGBoost. Si la instancia queda sin memoria, usa `t3.small` (2 GB).
> Ambas están disponibles en cuentas de estudiante.

---

### Paso 2: Conectarse y preparar el servidor

```bash
# Desde tu máquina local
chmod 400 tu-key.pem
ssh -i tu-key.pem ubuntu@<EC2_PUBLIC_IP>
```

```bash
# En el servidor EC2 — actualizar e instalar dependencias
sudo apt update && sudo apt upgrade -y

# Python 3.11 + herramientas
sudo apt install -y python3.11 python3.11-venv python3.11-dev \
  python3-pip git nginx certbot python3-certbot-nginx curl \
  libgomp1 build-essential

# Verificar Python
python3.11 --version
```

---

### Paso 3: Instalar PostgreSQL en el mismo EC2

```bash
# Instalar PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# Iniciar y habilitar
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Crear usuario y base de datos
sudo -u postgres psql << 'EOF'
CREATE USER medai WITH PASSWORD 'CAMBIA_ESTA_CONTRASEÑA';
CREATE DATABASE medai OWNER medai;
GRANT ALL PRIVILEGES ON DATABASE medai TO medai;
\q
EOF

# Verificar conexión
psql -h localhost -U medai -d medai -c "SELECT version();"
```

> PostgreSQL solo escucha en `localhost` por defecto — no está expuesto a Internet.
> No es necesario abrir el puerto 5432 en el Security Group.

---

### Paso 4: Desplegar el backend

```bash
# Clonar repositorio
sudo mkdir -p /app && sudo chown ubuntu:ubuntu /app
git clone https://github.com/TU_USUARIO/MedAI_Backend.git /app/MedAI_Backend
cd /app/MedAI_Backend

# Crear entorno virtual
python3.11 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

# Configurar entorno
cat > .env << EOF
DATABASE_URL=postgresql://medai:CAMBIA_ESTA_CONTRASEÑA@localhost/medai
CORS_ORIGINS=https://TU_DOMINIO_AMPLIFY.amplifyapp.com
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=480
EOF

# Entrenar el modelo (~5-15 min en t2.micro)
venv/bin/python train_model.py
```

---

### Paso 5: Crear el servicio systemd para el backend

```bash
sudo tee /etc/systemd/system/medai-backend.service << 'EOF'
[Unit]
Description=MedAI FastAPI Backend
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/app/MedAI_Backend
ExecStart=/app/MedAI_Backend/venv/bin/uvicorn api.main:app \
          --host 127.0.0.1 --port 8000 --workers 1
Restart=always
RestartSec=5
Environment=LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable medai-backend
sudo systemctl start medai-backend

# Verificar que está corriendo
sudo systemctl status medai-backend
curl http://localhost:8000/health
```

> Si el `LD_PRELOAD` da error, prueba con:
> `Environment=LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1`
> (depende de si tu EC2 es ARM o x86)

---

### Paso 6: Configurar Nginx con HTTPS

Necesitas un dominio. Si no tienes uno, puedes usar el **IP pública** del EC2 con HTTP
(sin HTTPS) solo para hacer pruebas — Nginx igualmente es útil como proxy.

#### Opción A: Con dominio y HTTPS (recomendado)

Puedes usar un dominio gratuito de [Freenom](https://www.freenom.com) o
[DuckDNS](https://www.duckdns.org) apuntando a tu IP de EC2.

```bash
# Configurar Nginx
sudo tee /etc/nginx/sites-available/medai << 'EOF'
server {
    listen 80;
    server_name api.tudominio.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/medai /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Obtener certificado SSL gratuito con Let's Encrypt
sudo certbot --nginx -d api.tudominio.com
```

#### Opción B: Sin dominio, solo IP (para pruebas académicas)

```bash
sudo tee /etc/nginx/sites-available/medai << 'EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/medai /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl enable nginx && sudo systemctl restart nginx
```

La API queda en `http://<EC2_PUBLIC_IP>/predict`

---

### Paso 7: Desplegar el frontend con AWS Amplify

Amplify es gratuito en cuentas de estudiante y es la forma más sencilla de servir Next.js en AWS.

1. Sube `MedAI_Frontend` a GitHub (repositorio separado)
2. En la consola de AWS → **AWS Amplify** → **New app → Host web app**
3. Conecta GitHub → selecciona el repo `MedAI_Frontend` → rama `main`
4. En **Build settings**, Amplify detecta Next.js automáticamente. Confirma:
   ```yaml
   version: 1
   frontend:
     phases:
       preBuild:
         commands:
           - npm install
       build:
         commands:
           - npm run build
     artifacts:
       baseDirectory: .next
       files:
         - '**/*'
     cache:
       paths:
         - node_modules/**/*
   ```
5. En **Environment variables** añade:
   ```
   NEXT_PUBLIC_API_URL  = https://api.tudominio.com
                          (o http://<EC2_IP> si usas Opción B sin dominio)
   NEXT_PUBLIC_USE_MOCK_API = false
   ```
6. Clic **Save and deploy** — Amplify te da una URL `*.amplifyapp.com`

7. Copia esa URL de Amplify y actualiza `CORS_ORIGINS` en el `.env` del EC2:
   ```bash
   # En el EC2
   nano /app/MedAI_Backend/.env
   # Cambia: CORS_ORIGINS=https://XXXXXXX.amplifyapp.com
   sudo systemctl restart medai-backend
   ```

---

### Paso 8: Verificar el despliegue

```bash
# Desde tu máquina local

# Health check
curl https://api.tudominio.com/health

# Registrar un usuario
curl -X POST https://api.tudominio.com/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","email":"demo@medai.local","password":"demo123","role":"nurse"}'

# Login
TOKEN=$(curl -s -X POST https://api.tudominio.com/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=demo&password=demo123" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Predicción completa
curl -X POST https://api.tudominio.com/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"edad":65,"sexo":"M","modo_llegada":"Ambulancia",
       "motivo_consulta":"Dolor en el pecho",
       "signos_vitales":{"frecuencia_cardiaca":110,"frecuencia_respiratoria":24,
         "temperatura":37.2,"spO2":88,"presion_sistolica":90,
         "presion_diastolica":60,"dolor":8},
       "comorbilidades":{"hta":true,"cardiopatia":true}}'

# Ver logs del backend
sudo journalctl -u medai-backend -f --no-pager
```

---

### Resumen de servicios AWS utilizados

| Servicio | Uso | Disponible en cuenta estudiante |
|----------|-----|---------------------------------|
| EC2 t2.micro | Backend + PostgreSQL + Nginx | ✅ Free Tier |
| AWS Amplify | Frontend Next.js | ✅ Gratuito |
| EC2 Key Pair | SSH | ✅ |
| Security Groups | Firewall | ✅ |
| ~~RDS~~ | ~~Base de datos gestionada~~ | ❌ No necesario |
| ~~CloudFront~~ | ~~CDN~~ | ❌ No necesario (Amplify lo incluye) |

---

## Model Retraining

```bash
# En el servidor EC2
cd /app/MedAI_Backend
venv/bin/python train_model.py
sudo systemctl restart medai-backend
```

**Métricas objetivo (paper):**
- Recall Nivel 1 ≥ 95% | Recall Nivel 2 ≥ 95% | F1 ponderado ≥ 80%

**Métricas actuales (dataset Chocó 100k):**
- Recall Nivel 1: 82.1% | Recall Nivel 2: 64.4% | F1: 52.6%

La diferencia se debe a la alta superposición de clases en el dataset balanceado de
15 variables. El diseño human-in-the-loop del sistema (clinician validation via SHAP)
compensa esta limitación del modelo base.

---

## Troubleshooting

### El backend no inicia en EC2
```bash
# Ver error completo
sudo journalctl -u medai-backend -n 50 --no-pager

# Probar manualmente para ver el error
cd /app/MedAI_Backend && venv/bin/python -c "from api.main import app; print('OK')"

# Modelo no existe
venv/bin/python train_model.py
```

### PostgreSQL: "connection refused" en EC2
```bash
sudo systemctl status postgresql
sudo systemctl restart postgresql
# Verificar que el usuario existe
sudo -u postgres psql -c "\du"
```

### libgomp error en EC2 al iniciar
```bash
# Encontrar el path correcto de libgomp
find /usr/lib -name "libgomp*.so*" 2>/dev/null
# Actualizar LD_PRELOAD en el servicio systemd con el path encontrado
sudo systemctl edit medai-backend
```

### Frontend en Amplify no puede llamar al backend (CORS)
```bash
# En el EC2, editar .env y agregar la URL de Amplify
nano /app/MedAI_Backend/.env
# CORS_ORIGINS=https://main.XXXX.amplifyapp.com
sudo systemctl restart medai-backend
```

### t2.micro sin memoria (Out of Memory)
```bash
# Crear swap de 1 GB para aguantar el entrenamiento del modelo
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### macOS: SHAP import error
```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib:/opt/homebrew/opt/libomp/lib
```
