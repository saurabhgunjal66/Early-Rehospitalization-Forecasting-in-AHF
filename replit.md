# CardioGuard AI — AHF Rehospitalization Predictor

## Overview
A clinical decision support platform for predicting 30-day AHF (Acute Heart Failure) rehospitalization risk, combining CNN-based medical imaging analysis with traditional ML models in an attractive dark-themed Streamlit interface.

## Architecture

### ML Models
- **CNN (`cnn_model.py`)**: Custom convolutional neural network built with numpy + scipy. Pipeline: Conv2D (4 filters, 3×3) → ReLU → MaxPool (4×4) → Texture Features → Logistic Classifier. Classifies 6 cardiac/pulmonary findings from medical images. Lazy-initialized on first use.
- **XGBoost**: Gradient boosting on clinical tabular features (14 variables)
- **Logistic Regression**: Clinical tabular model with SMOTE balancing
- **Ensemble**: Weighted blend of CNN (20%) + clinical models (80%)

### Pages
1. **Risk Assessment** — clinical form + ensemble prediction + SHAP explanation; integrates live CNN scan result
2. **Medical Imaging (CNN)** — image upload, CNN analysis, findings + probabilities + activation map + recommendations
3. **Patient Monitoring** — time-series dashboard, risk distribution, biomarker trends
4. **Model Performance** — ROC curves, confusion matrices, feature importance, drift monitoring
5. **Alerts & Notifications** — configurable thresholds, email settings, alert history
6. **Reports** — PDF/CSV/Excel report generation
7. **System Administration** — DB stats, user management, model retraining (Admin only)

### Backend Modules
- `auth.py` — user authentication & registration
- `database.py` — SQLite DB manager (assessments, image_scans, model_performance, patient_history, system_logs, data_quality_metrics tables)
- `data_generator.py` — synthetic training data generation
- `explainability.py` — SHAP-like feature contribution analysis
- `monitoring.py` — model drift detection, ROC comparison
- `notifications.py` — email alert system
- `alert_system.py` — automated clinical alert workflows
- `reporting.py` — PDF/CSV/Excel report generation
- `data_validation.py` — clinical data validation

## Run Command
```
uv run streamlit run app.py --server.port 5000 --server.address 0.0.0.0
```

## Key Dependencies
- streamlit>=1.50.0
- numpy, scipy (CNN convolutions)
- scikit-learn (classifiers, SMOTE)
- xgboost
- Pillow (image processing)
- plotly (interactive charts)
- pandas, reportlab, openpyxl

## Design Theme
Deep navy dark theme (#0d1b2a background), medical blue accents (#1e88e5), glassmorphism cards, gradient hero banners, custom CSS throughout.
