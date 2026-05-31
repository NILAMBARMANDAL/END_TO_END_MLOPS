# Predictive Maintenance — End-to-End MLOps

Predicts industrial device failure from IoT sensor telemetry.

**Stack:** MongoDB · ZenML · MLflow · scikit-learn (Random Forest) · FastAPI · Docker ·
Prometheus · Grafana · GitHub Actions CI · React

## Architecture
```
MongoDB Atlas ──> ZenML pipeline (ingest → clean → train → evaluate → GATE → deploy)
                                       │                         │
                                   MLflow (track)         FastAPI /predict ──> React UI
                                                               │
                                                  Prometheus ─> Grafana (monitoring)
```

## Quick start
See **GUIDE.md → Part 5** for the exact run order. TL;DR:
```bash
pip install -r requirements.txt
python run_deployment.py --config deploy --min-f1 0.6   # train + gated deploy
python export_model.py                                  # export model.pkl
uvicorn app:app --reload                                # serve API
docker compose up --build                               # full stack + monitoring
```

## Key design points
- Extreme class imbalance (106 failures / 124k rows) → under-sampling + F1-based gating.
- Config-driven model + hyperparameters via `config.yaml` (no code edits to retrain).
- Deployment gate promotes a model only if F1 ≥ floor AND ≥ previous best.
- Feature parity between training and serving prevents train/serve skew.

Full explanation, bug-fix log, interview Q&A, and limitations: **GUIDE.md**.
