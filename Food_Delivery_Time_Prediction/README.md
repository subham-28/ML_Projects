# 🍔 Food Delivery Time Prediction ⏱️

This project predicts the delivery time for food orders based on various real-world conditions like traffic, weather, and order details. It uses a trained **Stacking Regressor** (Random Forest + LGBM + Linear Regression) and is deployed via **FastAPI** with a responsive UI.

---

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version">
  </p>

---

## 📖 Table of Contents

- [About Project](#-about-project)
- [🚀 Features](#-features)
- [📁 Project Structure](#-project-structure)
- [📊 Model Overview](#-model-overview)
- [✅ Performance Metrics](#-performance-metrics)
- [🛠️ Installation](#️-installation)
- [💻 Tech Stack](#️-tech-stack)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 🚀 Features

- 🧠 Machine Learning powered predictions
- 🧩 Stacking Regressor for enhanced accuracy
- 🔎 Interactive Web UI with dark/light theme toggle
- 📈 Accurate delivery time predictions
- 🧪 Input validation with intelligent error handling
- 📊 MAE under ~3.1 minutes (Cross-validated)
- 📊 MLflow tracking via DagsHub
- 🌍 Deployment-ready using FastAPI

---

## 📁 Project Structure
```md
Food_Delivery_Time_Prediction/
├── api/
│ ├── main.py # FastAPI entrypoint
├── models/
│ └── stacking_model.pkl # Trained ML model (tracked via Git LFS)
│ └── preprocessing_pipeline.pkl 
├── src/
│ ├── inference.py # Prediction pipeline
│ └── data_preparation.py # Data Preprocessing
│ └── data_clean_utils.py # Cleaning utilities
├── data/
│ ├── swiggy.csv # Original Dataset
├── templates/
│ ├── index.html # Frontend html+css
├── requirements.txt # All dependencies
├── .gitattributes # Git LFS configuration
└── README.md # Project overview
```


---

## 📊 Model Overview

- Preprocessing using `ColumnTransformer`
- PowerTransformer used to normalize target
- Features used: 
  - Age, ratings, pickup time, distance
  - Categorical: traffic, weather, order type, etc.
- Model: `TransformedTargetRegressor` + `StackingRegressor` with:
  - RandomForest
  - LGBM
  - Linear Regression as meta-learner

### ✅ Performance Metrics

| Metric       | Value     |
|--------------|-----------|
| Train MAE    | ~2.4 min  |
| Test MAE     | ~3.1 min  |
| CV Score     | ~3.07 min |
| R² Score     | ~0.91     |

---

## 🛠️ Installation

```bash
# 1. Clone the repo
git clone https://github.com/subham-28/ML_Projects.git
cd ML_Projects/Food_Delivery_Time_Prediction

# 2. Model tracked via Git LFS
git lfs install
git lfs pull
git lfs track "*.pkl"

# 3. Set up environment (recommended using venv or conda):
pip install -r requirements.txt

# 4. Start FastAPI app:
uvicorn app.main:app --reload
```

---

## 💻 Technologies Stack

| Layer            | Technology                               |
|------------------|------------------------------------------|
| **Backend**  | [FastAPI](https://fastapi.tiangolo.com/) – High-performance API framework |
| **Modeling** | Scikit-learn, LightGBM, RandomForest, StackingRegressor           |
| **Preprocessing** | PowerTransformer, ColumnTransformer, Pipelines, OneHotEncoder, OrdinalEncoder |
| **Experiment Tracking** | [MLflow](https://mlflow.org/) with [DagsHub](https://dagshub.com/) integration |
| **Data Visualization** | Matplotlib, Seaborn, Plotly, Missingno (for missing value visualizations)      |
| **Data Storage** | Pandas, CSV (for static data)|
| **Deployment** | Uvicorn (Local Dev)     |
| **Version Control** | Git, GitHub, Git LFS (for large model files)               |

---

## 🤝 Contributing

1. Fork the repo
2. Create a new branch
3. Make your changes
4. Submit a pull request!

---

## 📜 License
MIT License. Feel free to use, modify, and share!

---

## 🙌 Acknowledgements
* DagsHub for free MLflow tracking
* FastAPI for rapid backend development
* scikit-learn & LightGBM for ML modeling
