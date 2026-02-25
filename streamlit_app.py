"""
Application Streamlit pour la détection de fraudes financières
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Détection de Fraude Financière",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CONSTANTES
# ============================================
MISSING_VAL = -999

# ============================================
# FONCTIONS UTILITAIRES
# ============================================
@st.cache_resource
def load_models():
    """Charge les modèles entraînés"""
    try:
        with open('../models/fraud_detection_models.pkl', 'rb') as f:
            models_dict = pickle.load(f)
        return models_dict
    except FileNotFoundError:
        st.error("❌ Modèles non trouvés. Veuillez d'abord entraîner les modèles.")
        return None

def preprocess_transaction(transaction_data):
    """
    Prétraite une transaction pour la prédiction
    """
    # Convertir en DataFrame si nécessaire
    if isinstance(transaction_data, dict):
        df = pd.DataFrame([transaction_data])
    else:
        df = transaction_data.copy()
    
    # Remplir les valeurs manquantes
    df = df.fillna(MISSING_VAL)
    
    # Feature engineering basique
    # (Ajoutez ici les mêmes transformations que lors de l'entraînement)
    
    # Exemple : hour
    if 'TransactionDT' in df.columns:
        df['hour'] = ((df['TransactionDT'] // 3600) % 24).astype(int)
    
    # Exemple : alertFeature
    if 'hour' in df.columns:
        def get_alert_level(hour):
            if hour in [4, 5, 6, 7, 8, 9]:
                return 3
            elif hour in [10, 11]:
                return 2
            elif hour in [0, 1, 2, 3, 16, 17, 18, 19, 20, 21, 22, 23]:
                return 1
            else:
                return 0
        df['alertFeature'] = df['hour'].apply(get_alert_level)
    
    return df

def predict_fraud(models_dict, transaction_df):
    """
    Fait une prédiction de fraude
    """
    try:
        # Récupérer les modèles
        xgb_models = models_dict['xgb_models']
        cat_models = models_dict['cat_models']
        stack_models = models_dict['stack_models']
        
        # Moyenne des prédictions de tous les folds
        xgb_preds = np.mean([m.predict_proba(transaction_df)[:, 1] for m in xgb_models], axis=0)
        cat_preds = np.mean([m.predict_proba(transaction_df)[:, 1] for m in cat_models], axis=0)
        
        # Stacking
        stack_features = np.column_stack([xgb_preds, cat_preds])
        final_preds = np.mean([m.predict_proba(stack_features)[:, 1] for m in stack_models], axis=0)
        
        return {
            'fraud_probability': float(final_preds[0]),
            'is_fraud': bool(final_preds[0] > 0.5),
            'xgb_score': float(xgb_preds[0]),
            'cat_score': float(cat_preds[0])
        }
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {str(e)}")
        return None

# ============================================
# INTERFACE PRINCIPALE
# ============================================
def main():
    # Header
    st.title("🔍 Détection de Fraudes Financières")
    st.markdown("### Système de détection basé sur l'Intelligence Artificielle")
    st.markdown("---")
    
    # Charger les modèles
    models_dict = load_models()
    
    if models_dict is None:
        st.stop()
    
    # Afficher les métriques du modèle
    st.sidebar.title("📊 Performance du Modèle")
    metrics = models_dict.get('metrics', {})
    if 'stack' in metrics:
        st.sidebar.metric("AUC", f"{metrics['stack']['auc']:.3f}")
        st.sidebar.metric("F1-Score", f"{metrics['stack']['f1']:.3f}")
        st.sidebar.metric("Precision", f"{metrics['stack'].get('precision', 0):.3f}")
        st.sidebar.metric("Recall", f"{metrics['stack'].get('recall', 0):.3f}")
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Modèle:** Stacking (XGBoost + CatBoost + LightGBM)")
    
    # Tabs pour différentes fonctionnalités
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Analyse Unitaire", 
        "📊 Analyse par Lot", 
        "📈 Statistiques",
        "ℹ️ À propos"
    ])
    
    # ============================================
    # TAB 1 : ANALYSE UNITAIRE
    # ============================================
    with tab1:
        st.header("Analyse d'une transaction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Saisie Manuelle")
            
            # Formulaire de saisie
            with st.form("transaction_form"):
                transaction_amt = st.number_input(
                    "Montant de la transaction ($)",
                    min_value=0.0,
                    value=100.0,
                    step=0.01
                )
                
                product_cd = st.selectbox(
                    "Type de produit",
                    ["W", "C", "R", "H", "S"]
                )
                
                card1 = st.number_input(
                    "Card1 (Identifiant carte)",
                    min_value=0,
                    value=13926
                )
                
                card2 = st.number_input(
                    "Card2",
                    min_value=0,
                    value=0
                )
                
                addr1 = st.number_input(
                    "Address1 (Code postal)",
                    min_value=0,
                    value=315
                )
                
                col_a, col_b = st.columns(2)
                with col_a:
                    p_emaildomain = st.text_input(
                        "Domaine email acheteur",
                        value="gmail.com"
                    )
                with col_b:
                    r_emaildomain = st.text_input(
                        "Domaine email destinataire",
                        value="gmail.com"
                    )
                
                device_type = st.selectbox(
                    "Type d'appareil",
                    ["desktop", "mobile", ""]
                )
                
                submit_button = st.form_submit_button("🔍 Analyser la transaction")
            
            if submit_button:
                # Créer la transaction
                transaction = {
                    'TransactionAmt': transaction_amt,
                    'ProductCD': product_cd,
                    'card1': card1,
                    'card2': card2,
                    'addr1': addr1,
                    'P_emaildomain': p_emaildomain,
                    'R_emaildomain': r_emaildomain,
                    'DeviceType': device_type,
                    'TransactionDT': int(datetime.now().timestamp())  # Timestamp actuel
                }
                
                # Prétraiter
                df = preprocess_transaction(transaction)
                
                # Prédire
                with st.spinner("Analyse en cours..."):
                    result = predict_fraud(models_dict, df)
                
                if result:
                    # Stocker dans session state pour affichage dans col2
                    st.session_state['last_result'] = result
                    st.session_state['last_transaction'] = transaction
                    st.rerun()
        
        with col2:
            st.subheader("📊 Résultat de l'Analyse")
            
            if 'last_result' in st.session_state:
                result = st.session_state['last_result']
                transaction = st.session_state['last_transaction']
                
                # Score de fraude
                fraud_prob = result['fraud_probability'] * 100
                
                # Jauge de risque
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=fraud_prob,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Score de Fraude (%)"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if fraud_prob > 50 else "green"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Verdict
                if result['is_fraud']:
                    st.error("🚨 **TRANSACTION SUSPECTE DÉTECTÉE**")
                    st.warning(f"**Probabilité de fraude : {fraud_prob:.2f}%**")
                    st.markdown("⚠️ Cette transaction présente des caractéristiques frauduleuses.")
                else:
                    st.success("✅ **TRANSACTION LÉGITIME**")
                    st.info(f"**Probabilité de fraude : {fraud_prob:.2f}%**")
                    st.markdown("✓ Cette transaction semble normale.")
                
                # Détails techniques
                with st.expander("🔬 Détails Techniques"):
                    st.markdown("**Scores des modèles individuels:**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("XGBoost", f"{result['xgb_score']*100:.2f}%")
                    with col_b:
                        st.metric("CatBoost", f"{result['cat_score']*100:.2f}%")
                    
                    st.markdown("**Informations de la transaction:**")
                    st.json(transaction)
            else:
                st.info("👈 Veuillez remplir le formulaire et cliquer sur 'Analyser'")
    
    # ============================================
    # TAB 2 : ANALYSE PAR LOT
    # ============================================
    with tab2:
        st.header("Analyse par lot de transactions")
        
        # Upload de fichier
        uploaded_file = st.file_uploader(
            "📤 Chargez un fichier CSV contenant les transactions",
            type=['csv'],
            help="Le fichier doit contenir les colonnes : TransactionAmt, ProductCD, card1, addr1, etc."
        )
        
        if uploaded_file is not None:
            try:
                # Charger le CSV
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Fichier chargé : {len(df)} transactions")
                
                # Afficher un aperçu
                with st.expander("👀 Aperçu des données"):
                    st.dataframe(df.head(10))
                
                # Bouton d'analyse
                if st.button("🔍 Analyser toutes les transactions"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    for idx, row in df.iterrows():
                        # Prétraiter
                        transaction_df = preprocess_transaction(row.to_dict())
                        
                        # Prédire
                        result = predict_fraud(models_dict, transaction_df)
                        
                        if result:
                            results.append({
                                'Index': idx,
                                'Fraud_Probability': result['fraud_probability'],
                                'Is_Fraud': result['is_fraud']
                            })
                        
                        # Mise à jour de la progression
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Analyse en cours... {idx+1}/{len(df)}")
                    
                    # Résultats
                    results_df = pd.DataFrame(results)
                    df_results = pd.concat([df, results_df.set_index('Index')], axis=1)
                    
                    st.success("✅ Analyse terminée!")
                    
                    # Statistiques
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total transactions", len(df_results))
                    with col2:
                        n_frauds = df_results['Is_Fraud'].sum()
                        st.metric("Fraudes détectées", n_frauds)
                    with col3:
                        fraud_rate = (n_frauds / len(df_results)) * 100
                        st.metric("Taux de fraude", f"{fraud_rate:.2f}%")
                    
                    # Distribution des scores
                    fig = px.histogram(
                        df_results, 
                        x='Fraud_Probability',
                        color='Is_Fraud',
                        nbins=50,
                        title="Distribution des scores de fraude",
                        labels={'Fraud_Probability': 'Probabilité de fraude', 'count': 'Nombre'},
                        color_discrete_map={True: 'red', False: 'green'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tableau des résultats
                    st.subheader("📋 Résultats détaillés")
                    st.dataframe(
                        df_results.sort_values('Fraud_Probability', ascending=False),
                        use_container_width=True
                    )
                    
                    # Téléchargement des résultats
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger les résultats (CSV)",
                        data=csv,
                        file_name=f"fraud_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du fichier : {str(e)}")
    
    # ============================================
    # TAB 3 : STATISTIQUES
    # ============================================
    with tab3:
        st.header("📈 Statistiques du Modèle")
        
        # Matrice de confusion (exemple avec les résultats d'entraînement)
        st.subheader("Matrice de Confusion")
        
        cm_data = [[113565, 410], [813, 3320]]
        fig = px.imshow(
            cm_data,
            labels=dict(x="Prédit", y="Réel", color="Nombre"),
            x=['Non-Fraude', 'Fraude'],
            y=['Non-Fraude', 'Fraude'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Métriques détaillées
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Métriques de Performance")
            metrics_data = {
                'Métrique': ['AUC', 'F1-Score', 'Precision', 'Recall'],
                'Score': [0.979, 0.844, 0.890, 0.803]
            }
            fig = px.bar(
                metrics_data,
                x='Métrique',
                y='Score',
                text='Score',
                color='Score',
                color_continuous_scale='Viridis'
            )
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig.update_layout(showlegend=False, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⚖️ Analyse des Erreurs")
            error_data = {
                'Type': ['Vrais Positifs (TP)', 'Faux Positifs (FP)', 
                        'Vrais Négatifs (TN)', 'Faux Négatifs (FN)'],
                'Nombre': [3320, 410, 113565, 813]
            }
            fig = px.pie(
                error_data,
                values='Nombre',
                names='Type',
                title='Répartition des Prédictions'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # TAB 4 : À PROPOS
    # ============================================
    with tab4:
        st.header("ℹ️ À propos du projet")
        
        st.markdown("""
        ### 🎓 Projet Académique
        
        **Détection de Fraudes Financières à partir des Transactions à l'aide de l'Intelligence Artificielle**
        
        #### 👥 Réalisé par
        - **Farah REBAI**
        - **Lobna ELABED**
        
        #### 👨‍🏫 Encadré par
        - **M. Riadh ABDELFATTAH**
        
        #### 🏫 Institution
        - École Supérieure des Communications de Tunis (Sup'Com)
        - Université de Carthage
        - Année universitaire : 2024-2025
        
        ---
        
        ### 🤖 Architecture du Modèle
        
        Le système utilise une approche de **Stacking** combinant :
        1. **XGBoost** - Gradient Boosting optimisé
        2. **CatBoost** - Gestion native des variables catégorielles
        3. **LightGBM** - Meta-modèle pour prédiction finale
        
        ### 📊 Dataset
        - **Source** : IEEE-CIS Fraud Detection (Kaggle)
        - **Transactions** : ~590,000
        - **Features** : 400+ variables
        - **Déséquilibre** : ~3.5% de fraudes
        
        ### 🎯 Performances
        - **AUC** : 97.9%
        - **F1-Score** : 84.4%
        - **Precision** : 89.0%
        - **Recall** : 80.3%
        
        ### 🚀 Technologies
        - Python 3.8+
        - Scikit-learn
        - XGBoost, CatBoost, LightGBM
        - Streamlit
        - Plotly
        
        ### 📝 Licence
        Projet académique - Sup'Com 2024-2025
        """)
        
        st.markdown("---")
        st.info("💡 **Note** : Ce système est conçu à des fins éducatives et de démonstration.")

# ============================================
# POINT D'ENTRÉE
# ============================================
if __name__ == "__main__":
    main()