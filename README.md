<div align="center">

<!-- Logo / Banner -->
<img src="https://img.shields.io/badge/Gravity_AI-Phoenix_Cyber_Security-0a0a0a?style=for-the-badge&logo=shield&logoColor=00f5ff" />

# 🚗 ADAS Phantom Braking Detection
### MLOps Pipeline · Sensor Fusion Intelligence · Ghost Obstacle Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Integrated-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Config-Driven](https://img.shields.io/badge/Config-YAML_Driven-4CAF50?style=flat-square&logo=yaml&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)]()

> **Developed by [Gravity AI · Phoenix Cyber Security]()**  
> *Intelligent Vehicle Safety · Autonomous Systems Division*

</div>

---

## 📌 Overview

This project builds a **production-grade machine learning pipeline** to detect **phantom braking scenarios** in Advanced Driver Assistance Systems (ADAS).

**Phantom braking** occurs when a vehicle's Automatic Emergency Braking (AEB) system triggers a hard stop despite **no real obstacle** being present — caused by sensor noise, environmental interference, or ghost radar targets.

### Goals

| Objective | Description |
|-----------|-------------|
| 🎯 Ghost Detection | Identify false-positive obstacle detections from sensor fusion |
| 🛑 Reduce False AEB | Minimize unnecessary emergency braking events |
| 🔬 Simulate Sensor Logic | Replicate real-world multi-sensor decision making |

### Pipeline Capabilities

- ✅ Configuration-driven execution via `config.yaml`
- ✅ Modular training, evaluation, and inference modules
- ✅ Centralized structured logging
- ✅ Experiment reproducibility via seeded splits
- ✅ Plug-and-play model switching (no code changes required)

---

## 🚨 Problem Statement

ADAS systems fuse inputs from three primary sensor types:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    RADAR    │    │   CAMERA    │    │    LiDAR    │
│  Distance   │ +  │  Confidence │ +  │   Density   │  →  Decision
│  Velocity   │    │  BBox Area  │    │   Points    │
│    Angle    │    │  Obj. Type  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
```

### False Detection Sources

| Sensor | Ghost Cause Example |
|--------|---------------------|
| Radar | Reflections from overpasses, bridges, guard rails |
| Camera | Shadows, road markings, painted lines misclassified |
| LiDAR | Roadside vegetation, debris, dust particles |

### Classification Task

```
Input: Multi-sensor fusion feature vector
         ↓
   ML Classifier
         ↓
Output:  1 → Real Obstacle  (maintain AEB)
         0 → Phantom / Ghost  (suppress AEB)
```

---

## 📊 Dataset

Simulated sensor fusion inputs representing an ADAS decision environment (`50,000 samples`).

### Feature Reference

| Feature | Unit | Description |
|---------|------|-------------|
| `radar_distance` | meters | Distance from ego vehicle to detected object |
| `relative_velocity` | m/s | Closing speed; negative = approaching |
| `radar_angle` | degrees | Object bearing from vehicle centerline |
| `radar_rcs` | — | Radar Cross Section; reflection strength |
| `object_persistence` | frames | Consecutive frames object is tracked |
| `trajectory_smoothness` | [0–1] | Stability of object motion path |
| `camera_confidence` | [0–1] | Camera model detection score |
| `object_type` | encoded | 0=unknown · 1=vehicle · 2=pedestrian |
| `bbox_area` | px² | Bounding box area; larger = closer object |
| `lidar_density` | [0–1] | LiDAR point cloud density in cluster |
| `ego_speed` | km/h | Ego vehicle speed |
| `ego_acceleration` | m/s² | Longitudinal acceleration; negative = braking |
| `steering_angle` | rad | Wheel angle; ~0 = straight line |
| `time_to_collision` | seconds | Estimated TTC; < 1.5s triggers AEB |
| `lane_overlap` | [0–1] | Object overlap with ego driving lane |
| `sensor_agreement` | [0–1] | Consensus across radar, camera, LiDAR |
| `rain_intensity` | [0–1] | Precipitation level affecting sensors |
| `lighting_level` | [0–1] | Ambient light affecting camera |

### Target Variable

```yaml
real_obstacle:
  1: Real physical obstacle — AEB should activate
  0: Phantom / ghost detection — AEB should be suppressed
```

---

## 🗂 Project Structure

```
adas-phantom-braking-mlops/
│
├── configs/
│   └── config.yaml              # Pipeline configuration (mode, paths, hyperparams)
│
├── data/
│   └── raw/
│       └── adas_phantom_braking_dataset_50k.csv
│
├── logs/
│   └── pipeline.log             # Runtime logs (auto-generated)
│
├── models/
│   └── best_model.pkl           # Saved model artifact (auto-generated)
│
├── src/
│   ├── data/
│   │   └── preprocess.py        # Feature engineering, train/test split
│   │
│   ├── models/
│   │   ├── train_model.py       # Model training logic
│   │   ├── evaluate_model.py    # Metrics: accuracy, F1, confusion matrix
│   │   └── predict_model.py     # Inference on test inputs
│   │
│   └── utils/
│       └── logger.py            # Centralized logging setup
│
├── main.py                      # Pipeline entrypoint
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone <repo_url>
cd adas-phantom-braking-mlops
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

### Training Mode

Set `mode` in `configs/config.yaml`:

```yaml
mode: train
```

Execute:

```bash
python main.py
```

> Trains the selected model, evaluates on the test split, and saves the artifact to `models/best_model.pkl`.

---

### Inference / Test Mode

```yaml
mode: test
```

```bash
python main.py
```

> Loads the saved model and runs prediction using the sensor parameters defined in `config.yaml` under the test input section.

---

## 🧠 Model Configuration

Models are fully configured via `config.yaml` — **no code changes needed**.

### Select Active Model

```yaml
model:
  selected_model: xgboost   # Options: random_forest | xgboost | logistic_regression
```

### Hyperparameter Reference

```yaml
models:
  random_forest:
    n_estimators: 200   # Trees in ensemble
    max_depth: 10       # Max depth per tree

  xgboost:
    n_estimators: 300   # Boosting rounds
    max_depth: 6        # Tree depth
    learning_rate: 0.1  # Shrinkage per step (η)

  logistic_regression:
    max_iter: 200       # Solver iteration limit
```

---

## 📋 Logging

All runtime events are written to:

```
logs/pipeline.log
```

| Log Event | Description |
|-----------|-------------|
| Pipeline Start | Mode, config snapshot |
| Preprocessing | Dataset shape, split sizes |
| Training | Model name, hyperparameters |
| Evaluation | Accuracy, F1-score, confusion matrix |
| Model Save | Output path confirmation |
| Prediction | Input features → output class |
| Errors | Exceptions with traceback |

---

## 🏗 MLOps Features

| Feature | Implementation |
|---------|----------------|
| Config-driven execution | `config.yaml` controls all pipeline behaviour |
| Modular architecture | Independent `preprocess`, `train`, `evaluate`, `predict` modules |
| Reproducibility | Fixed `random_state` seed across all splits and models |
| Centralized logging | Single logger injected across all modules |
| Deployable inference | `predict_model.py` accepts any sensor input dict |
| Model agnostic | Swap models via YAML with zero code changes |

---

## 🔭 Future Improvements

- [ ] **FastAPI service** — REST endpoint for real-time AEB decision inference
- [ ] **MLflow integration** — Experiment tracking, metric versioning, model registry
- [ ] **Real-time sensor simulation** — Streaming sensor data via Kafka / MQTT
- [ ] **ADAS scenario dashboard** — Visual replay of detection decisions
- [ ] **Docker deployment** — Containerized pipeline with `docker-compose`
- [ ] **CI/CD pipeline** — Automated retraining on new data ingestion

---

## 👥 Authors & Credits

<div align="center">

| | |
|---|---|
| **Organization** | Gravity AI · Phoenix Cyber Security |
| **Division** | Autonomous Systems & Intelligent Vehicle Safety |
| **Project** | ADAS Phantom Braking Detection — MLOps Pipeline |

</div>

---

<div align="center">

**© 2024 Gravity AI · Phoenix Cyber Security. All rights reserved.**

*Building safer roads through intelligent sensing.*

</div>