# 🛡️ Global Cybersecurity Threats (2015–2024) — Machine Learning Pipeline

> **MOD10 : Machine Learning** — Projet de fin de semestre  
> Prédiction des types d'attaques cybernétiques et estimation des pertes financières à l'aide de techniques avancées de Machine Learning.

---

## 📋 Table des matières

- [Aperçu du projet](#-aperçu-du-projet)
- [Dataset](#-dataset)
- [Architecture du projet](#-architecture-du-projet)
- [Feature Engineering](#-feature-engineering)
- [Modèles entraînés](#-modèles-entraînés)
- [Pipeline ML complète](#-pipeline-ml-complète)
- [Évaluation des modèles](#-évaluation-des-modèles)
- [Interprétabilité (SHAP & LIME)](#-interprétabilité-shap--lime)
- [Suivi MLflow](#-suivi-mlflow)
- [API REST (FastAPI)](#-api-rest-fastapi)
- [Déploiement Docker](#-déploiement-docker)
- [Installation & Exécution](#-installation--exécution)
- [Structure des fichiers](#-structure-des-fichiers)
- [Résultats](#-résultats)
- [Technologies utilisées](#-technologies-utilisées)

---

## 🎯 Aperçu du projet

Ce projet implémente une **pipeline complète de Machine Learning** pour analyser les menaces de cybersécurité mondiales entre 2015 et 2024. Il couvre deux tâches prédictives :

| Tâche              | Objectif                                                    | Type         |
| ------------------ | ----------------------------------------------------------- | ------------ |
| **Classification** | Prédire le type d'attaque (`Attack Type`)                   | Multi-classe |
| **Régression**     | Estimer la perte financière (`Financial Loss in Million $`) | Continue     |

Le pipeline comprend : EDA → Feature Engineering → Entraînement (avec Hyperparameter Tuning) → Évaluation → Interprétabilité (SHAP/LIME) → API REST → Docker.

---

## 📊 Dataset

- **Fichier** : `Global_Cybersecurity_Threats_2015-2024.csv`
- **Période** : 2015 – 2024
- **Colonnes d'entrée** (10) :

| Colonne                               | Type         | Description                           |
| ------------------------------------- | ------------ | ------------------------------------- |
| `Country`                             | Catégorielle | Pays de l'incident                    |
| `Year`                                | Numérique    | Année de l'attaque                    |
| `Attack Type`                         | Catégorielle | Type d'attaque (cible classification) |
| `Target Industry`                     | Catégorielle | Secteur ciblé                         |
| `Financial Loss (in Million $)`       | Numérique    | Perte financière (cible régression)   |
| `Number of Affected Users`            | Numérique    | Nombre d'utilisateurs affectés        |
| `Attack Source`                       | Catégorielle | Source de l'attaque                   |
| `Security Vulnerability Type`         | Catégorielle | Type de vulnérabilité                 |
| `Defense Mechanism Used`              | Catégorielle | Mécanisme de défense                  |
| `Incident Resolution Time (in Hours)` | Numérique    | Temps de résolution                   |

---

## 🏗️ Architecture du projet

```
Project/
├── main.py                          # Point d'entrée — exécute toute la pipeline
├── requirements.txt                 # Dépendances Python
├── Dockerfile                       # Image Docker pour l'API
├── docker-compose.yml               # Orchestration Docker
├── .dockerignore                    # Exclusions Docker (mlruns, __pycache__, etc.)
├── .gitignore
├── README.md
├── Global_Cybersecurity_Threats_2015-2024.csv  # Dataset
│
├── src/                             # Code source principal
│   ├── config.py                    # Configuration (chemins, features, constantes)
│   ├── logger.py                    # Logger rich (console colorée, barres de progression)
│   ├── preprocessing.py             # Chargement, nettoyage, feature engineering, save/load splits
│   ├── eda.py                       # Analyse exploratoire — 12 visualisations
│   ├── train_classification.py      # Entraînement classification (5 modèles)
│   ├── train_regression.py          # Entraînement régression (5 modèles)
│   ├── evaluate.py                  # Évaluation comparative
│   └── interpretability.py          # SHAP & LIME
│
├── api/                             # API REST FastAPI + Interface Web
│   ├── app.py                       # Endpoints de prédiction + CORS
│   ├── schemas.py                   # Schémas Pydantic (validation avec Literal types)
│   ├── templates/
│   │   └── index.html               # Interface web (dark/light theme, formulaires)
│   └── static/
│       └── style.css                # Styles CSS (thèmes, animations, responsive)
│
├── models/                          # Modèles sérialisés (.pkl)
├── plots/                           # Toutes les visualisations (.png)
└── mlruns/                          # Logs MLflow (métriques, params, artifacts)
```

---

## ⚙️ Feature Engineering

À partir des 10 colonnes brutes, **6 features dérivées** sont créées :

| Feature              | Formule                                                 | Justification                    |
| -------------------- | ------------------------------------------------------- | -------------------------------- |
| `Loss_per_User`      | `Financial Loss × 1M / Nb Users` (clippé au P99)        | Impact financier par utilisateur |
| `Users_per_Hour`     | `Nb Users / Resolution Time` (clippé au P99)            | Vitesse de propagation           |
| `Loss_per_Hour`      | `Financial Loss / Resolution Time` (clippé au P99)      | Efficience de résolution         |
| `Log_Financial_Loss` | `log(1 + Financial Loss)`                               | Réduction de l'asymétrie         |
| `Year_Period`        | Bins `[2015-2017, 2018-2020, 2021-2030]`                | Tendances temporelles            |
| `Attack_Severity`    | Bins sur Financial Loss `[Low, Medium, High, Critical]` | Catégorisation de la gravité     |

> Les features `Loss_per_User`, `Users_per_Hour` et `Loss_per_Hour` sont **clippées au 99e percentile** pour limiter l'impact des valeurs extrêmes.

**Préprocessing** : `StandardScaler` (numériques) + `OneHotEncoder` (catégorielles) via `ColumnTransformer`.

---

## 🤖 Modèles entraînés

### Classification (prédiction du type d'attaque)

| #   | Modèle                | Technique                                                                |
| --- | --------------------- | ------------------------------------------------------------------------ |
| 1   | **Random Forest**     | GridSearchCV (`n_estimators`, `max_depth`, `min_samples_*`)              |
| 2   | **XGBoost**           | GridSearchCV (`n_estimators`, `max_depth`, `learning_rate`, `subsample`) |
| 3   | **SVM (RBF)**         | GridSearchCV (`C`, `gamma`)                                              |
| 4   | **Voting Ensemble**   | Soft voting (RF + XGB + SVM)                                             |
| 5   | **Stacking Ensemble** | Base: RF + XGB + SVM → Meta: Logistic Regression, CV=5                   |

### Régression (estimation des pertes financières)

| #   | Modèle                | Technique                                  |
| --- | --------------------- | ------------------------------------------ |
| 1   | **Random Forest**     | GridSearchCV                               |
| 2   | **XGBoost**           | GridSearchCV                               |
| 3   | **Ridge Regression**  | GridSearchCV (`alpha`)                     |
| 4   | **Voting Ensemble**   | RF + XGB + Ridge                           |
| 5   | **Stacking Ensemble** | Base: RF + XGB + Ridge → Meta: Ridge, CV=5 |

> Tous les modèles utilisent **GridSearchCV** avec **5-fold cross-validation**.

---

## 🔄 Pipeline ML complète

```
python main.py
```

Exécute séquentiellement :

```
Step 1/5 → EDA               → 12 graphiques dans plots/
Step 2/5 → Classification     → 5 modèles + GridSearch + MLflow
Step 3/5 → Régression         → 5 modèles + GridSearch + MLflow
Step 4/5 → Évaluation         → Confusion matrices, ROC, résiduals
Step 5/5 → Interprétabilité   → SHAP (summary, bar, waterfall, dependence) + LIME
```

**Inputs** : `Global_Cybersecurity_Threats_2015-2024.csv` (données brutes)

**Outputs** :

- `models/` — Modèles sérialisés (`.pkl`), preprocessors, label encoders
- `plots/` — ~35 visualisations PNG
- `mlruns/` — Logs MLflow (params, métriques, artifacts, modèles)

---

## 📈 Évaluation des modèles

### Métriques de Classification

| Métrique                        | Description                         |
| ------------------------------- | ----------------------------------- |
| **Accuracy**                    | Taux de prédictions correctes       |
| **Precision** (weighted)        | Précision pondérée par classe       |
| **Recall** (weighted)           | Rappel pondéré par classe           |
| **F1-score** (weighted & macro) | Moyenne harmonique precision/recall |
| **CV Accuracy**                 | Accuracy moyenne sur 5 folds        |

### Métriques de Régression

| Métrique       | Description                  |
| -------------- | ---------------------------- |
| **R²**         | Coefficient de détermination |
| **MAE**        | Erreur absolue moyenne       |
| **MSE / RMSE** | Erreur quadratique moyenne   |
| **CV R²**      | R² moyen sur 5 folds         |

### Visualisations d'évaluation

- Matrices de confusion (5 modèles côte à côte)
- Courbes ROC/AUC (One-vs-Rest par classe)
- Graphiques Actual vs Predicted (régression)
- Graphiques de résidus
- Bar charts de comparaison

---

## 🔍 Interprétabilité (SHAP & LIME)

### SHAP (SHapley Additive exPlanations)

| Plot                | Description                                     | Fichier                                            |
| ------------------- | ----------------------------------------------- | -------------------------------------------------- |
| **Summary plot**    | Distribution des SHAP values par feature        | `shap_clf_summary.png`, `shap_reg_summary.png`     |
| **Bar plot**        | Importance moyenne des features                 | `shap_clf_bar.png`, `shap_reg_bar.png`             |
| **Waterfall plot**  | Décomposition de la prédiction d'un échantillon | `shap_clf_waterfall.png`, `shap_reg_waterfall.png` |
| **Dependence plot** | Relation feature ↔ SHAP value (top 3)           | `shap_reg_dependence_top*.png`                     |

### LIME (Local Interpretable Model-agnostic Explanations)

| Plot               | Description                          | Fichier                     |
| ------------------ | ------------------------------------ | --------------------------- |
| **Classification** | Explication locale de 3 échantillons | `lime_clf_sample_1/2/3.png` |
| **Régression**     | Explication locale de 3 échantillons | `lime_reg_sample_1/2/3.png` |

---

## 📊 Suivi MLflow

Deux expériences sont trackées :

| Expérience                     | Contenu loggé                                                                                           |
| ------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `Cybersecurity_Classification` | Params (hyperparams), métriques (accuracy, F1, CV), modèles sklearn, confusion matrices (JSON artifact) |
| `Cybersecurity_Regression`     | Params, métriques (R², MAE, RMSE, CV), modèles sklearn, résumés (JSON artifact)                         |

**Lancer l'interface MLflow :**

```bash
mlflow ui --backend-store-uri mlruns
```

Puis ouvrir : [http://localhost:5000](http://localhost:5000)

---

## 🌐 API REST (FastAPI) & Interface Web

L'API expose 4 endpoints et inclut une **interface web interactive** :

| Méthode | Endpoint                  | Description                                   |
| ------- | ------------------------- | --------------------------------------------- |
| `GET`   | `/`                       | Interface web (formulaires, thème dark/light) |
| `GET`   | `/health`                 | Vérification de santé + état des modèles      |
| `POST`  | `/predict/attack-type`    | Prédiction du type d'attaque + probabilités   |
| `POST`  | `/predict/financial-loss` | Estimation de la perte financière             |

### Interface web

- **Thème dark/light** avec bascule et sauvegarde `localStorage`
- **Formulaires** avec validation côté client et scénarios pré-remplis
- **Indicateurs de confiance** et labels de sévérité sur les prédictions
- **Notifications toast** avec animations
- **Timeout 10s** sur les requêtes (`AbortController`)
- **CORS** activé (`CORSMiddleware`)
- **Validation stricte** des champs catégoriels via `Literal` types Pydantic (retour 422 explicite)

### Exemple d'utilisation

```bash
# Health check
curl http://localhost:8000/health

# Prédiction de type d'attaque
curl -X POST http://localhost:8000/predict/attack-type \
  -H "Content-Type: application/json" \
  -d '{
    "Country": "France",
    "Year": 2023,
    "Target_Industry": "Banking",
    "Financial_Loss": 45.5,
    "Number_of_Affected_Users": 500000,
    "Attack_Source": "Hacker Group",
    "Security_Vulnerability_Type": "Weak Passwords",
    "Defense_Mechanism_Used": "Firewall",
    "Incident_Resolution_Time": 30.0
  }'

# Prédiction de perte financière
curl -X POST http://localhost:8000/predict/financial-loss \
  -H "Content-Type: application/json" \
  -d '{
    "Country": "France",
    "Year": 2023,
    "Attack_Type": "Ransomware",
    "Target_Industry": "Banking",
    "Number_of_Affected_Users": 500000,
    "Attack_Source": "Hacker Group",
    "Security_Vulnerability_Type": "Weak Passwords",
    "Defense_Mechanism_Used": "Firewall",
    "Incident_Resolution_Time": 30.0
  }'
```

### Documentation interactive

FastAPI génère automatiquement la documentation Swagger :

- **Swagger UI** : [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc** : [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🐳 Déploiement Docker

```bash
# Construire et lancer
docker-compose up --build

# L'API est accessible sur http://localhost:8000
```

Le `Dockerfile` utilise `python:3.10-slim`, copie les modèles pré-entraînés et lance `uvicorn`.
Un fichier `.dockerignore` exclut `mlruns/`, `__pycache__/`, `.git/`, `plots/`, `logs/` pour optimiser le build.

---

## 🚀 Installation & Exécution

### Prérequis

- Python 3.10+
- pip

### Installation

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd Project

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Exécution complète de la pipeline

```bash
python main.py
```

### Lancer uniquement l'API

```bash
# Prérequis : les modèles doivent être dans models/
python api/app.py
# ou
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Lancer avec Docker

```bash
docker-compose up --build
```

---

## 📁 Structure des fichiers

| Fichier/Dossier               | Rôle                                                              |
| ----------------------------- | ----------------------------------------------------------------- |
| `main.py`                     | Orchestrateur de la pipeline complète (5 étapes)                  |
| `src/config.py`               | Chemins, listes de features, constantes, config MLflow            |
| `src/logger.py`               | Logger `rich` (console colorée, barres de progression, tableaux)  |
| `src/preprocessing.py`        | Load → Clean → Feature Engineering → save/load splits             |
| `src/eda.py`                  | 12 visualisations exploratoires                                   |
| `src/train_classification.py` | 5 modèles de classification + GridSearchCV + MLflow               |
| `src/train_regression.py`     | 5 modèles de régression + GridSearchCV + MLflow                   |
| `src/evaluate.py`             | Évaluation comparative + graphiques                               |
| `src/interpretability.py`     | SHAP (summary, bar, waterfall, dependence) + LIME                 |
| `api/app.py`                  | API FastAPI (4 endpoints) + CORS + chargement robuste des modèles |
| `api/schemas.py`              | Schémas Pydantic avec validation `Literal` types                  |
| `api/templates/index.html`    | Interface web interactive (dark/light theme)                      |
| `api/static/style.css`        | Styles CSS (thèmes, animations, responsive)                       |
| `requirements.txt`            | Dépendances Python                                                |
| `Dockerfile`                  | Image Docker (`python:3.10-slim`)                                 |
| `docker-compose.yml`          | Orchestration Docker                                              |
| `.dockerignore`               | Exclusions pour le build Docker                                   |

---

## 📊 Résultats

> Les résultats ci-dessous sont générés automatiquement par la pipeline. Relancez `python main.py` pour mettre à jour.

### Visualisations générées (`plots/`)

| Catégorie                     | Nombre   | Exemples                                        |
| ----------------------------- | -------- | ----------------------------------------------- |
| **EDA**                       | 12       | Distributions, heatmaps, tendances temporelles  |
| **Évaluation Classification** | 7+       | Confusion matrices, ROC curves, comparaisons    |
| **Évaluation Régression**     | 4+       | Actual vs Predicted, résidus, comparaisons      |
| **SHAP**                      | 7+       | Summary, bar, waterfall, dependence plots       |
| **LIME**                      | 6        | 3 samples classification + 3 samples régression |
| **Total**                     | **~35+** |                                                 |

---

## 🛠️ Technologies utilisées

| Catégorie               | Technologie                      |
| ----------------------- | -------------------------------- |
| **Langage**             | Python 3.10                      |
| **ML / Modèles**        | scikit-learn, XGBoost            |
| **Feature Engineering** | pandas, NumPy                    |
| **Visualisation**       | Matplotlib, Seaborn              |
| **Interprétabilité**    | SHAP, LIME                       |
| **Tracking MLOps**      | MLflow                           |
| **API**                 | FastAPI, Uvicorn, Pydantic       |
| **Logging**             | Rich (console colorée, tableaux) |
| **Containerisation**    | Docker, Docker Compose           |

---

## 👤 Auteur

Projet réalisé dans le cadre du module **TI608 — Machine Learning** (S6, 2025–2026).
