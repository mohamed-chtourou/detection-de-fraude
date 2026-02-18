# 🔍 Détection de Fraudes Financières

Projet d'apprentissage automatique pour la détection de fraudes financières à partir de transactions bancaires, utilisant des techniques avancées de Machine Learning.

## 📊 Dataset

Dataset IEEE-CIS Fraud Detection de Kaggle contenant :
- Transactions réelles du commerce en ligne
- Données d'identité des utilisateurs
- Variables anonymisées pour la confidentialité

## 🚀 Technologies

- **Python 3.8+**
- **Scikit-learn** : Prétraitement et métriques
- **XGBoost** : Modèle de boosting
- **CatBoost** : Gestion native des catégorielles
- **LightGBM** : Meta-modèle (stacking)
- **Flask** : API de déploiement
- **Pandas & NumPy** : Manipulation des données

## 📁 Structure du projet
```
detection-de-fraude/
├── data/                          # Données (non versionnées)
│   ├── train_transaction.csv
│   └── train_identity.csv
├── notebooks/                     # Notebooks d'exploration
├── src/
│   ├── preprocessing.py          # Étape 1 & 2 : Nettoyage
│   ├── feature_engineering.py    # Étape 3 : Features
│   └── modeling.py               # Étape 4 : Entraînement
├── models/                        # Modèles sauvegardés
├── figures/                       # Visualisations
├── app/                          # Application Flask
│   ├── app.py
│   ├── templates/
│   └── static/
├── requirements.txt
└── README.md
```

## 🔧 Installation
```bash
# Cloner le repo
git clone https://github.com/mohamed-chtourou/detection-de-fraude.git
cd detection-de-fraude

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## 📈 Pipeline du projet

### 1️⃣ Prétraitement des données
```bash
python src/preprocessing_step1.py
python src/preprocessing_step2.py
```

### 2️⃣ Feature Engineering
```bash
python src/feature_engineering.py
```

### 3️⃣ Entraînement du modèle
```bash
python src/modeling.py
```

### 4️⃣ Déploiement
```bash
python app/app.py
```

## 📊 Résultats

- **AUC** : ~0.96-0.98
- **F1-Score** : ~0.85-0.90
- **Précision** : ~0.88-0.92
- **Recall** : ~0.85-0.88

## 🎯 Features clés

### Features temporelles
- `alertFeature` : Classification de criticité par heure
- `uid_timedelta` : Temps entre transactions
- `first_tran` : Temps depuis première transaction

### Features utilisateur (UID)
- Agrégations par utilisateur (moyenne, écart-type)
- Taux de transactions par UID
- Détection d'anomalies (z-score)

### Features appareil
- Hash d'appareil unique
- Nombre d'appareils par utilisateur
- Nombre d'utilisateurs par appareil

### Features comportementales
- `open_card` : Âge du compte
- `decimal_digit` : Précision du montant
- `email_domain_comp` : Comparaison domaines email

## 👥 Auteurs

MOHAMED CHTOUROU
AMIR ABBES



**École** : École Supérieure des Communications de Tunis (Sup'Com)

**Année universitaire** : 2024-2025

## 📄 Licence

Ce projet est réalisé dans le cadre d'un projet académique.
