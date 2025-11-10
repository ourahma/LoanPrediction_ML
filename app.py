import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="🏦",
    layout="wide"
)

# Titre de l'application
st.title("🏦 Prediction system is ready")
st.markdown("""
This project uses the **Ensemble learning** to predict loan probability using multiple machine learning algorithme.
""")

# Sidebar pour les informations
st.sidebar.header("ℹ️ Info")
st.sidebar.markdown("""
**Available models:**
- Random Forest
- KNN
- Logistic regression  
- SVM
- Gradient Boosting
- Voting Classifier (Hard)
- Voting Classifier (Soft)
- Stacking Classifier
""")

# Fonction de prétraitement identique à l'entraînement
def preprocess_input_data(input_df, label_encoders, scaler, numerical_columns, categorical_columns):
    """
    Pre-traiement des données.
    """
    data = input_df.copy()
    
    # Encodage des variables catégorielles
    for col in categorical_columns:
        if col in data.columns and col in label_encoders:
            # Remplir les valeurs manquantes
            data[col] = data[col].fillna("Unknown") if data[col].dtype == "object" else data[col]
            
            # Gérer les nouvelles valeurs non vues pendant l'entraînement
            try:
                data[col] = label_encoders[col].transform(data[col])
            except ValueError:
                # Si nouvelle valeur, lui assigner -1 (valeur inconnue)
                st.warning(f"Missed value detected {col}. Using default value.")
                data[col] = -1
    
    # Imputation des valeurs numériques
    numeric_imputer = SimpleImputer(strategy="median")
    data[numerical_columns] = numeric_imputer.fit_transform(data[numerical_columns])
    
    # Standardisation
    data[numerical_columns] = scaler.transform(data[numerical_columns])
    
    return data

# Chargement des modèles et préprocesseurs
@st.cache_resource
def load_models_and_preprocessors():
    """
    Load models
    """
    try:
        models = {
            'Random Forest': joblib.load('models/rf.joblib'),
            'KNN': joblib.load('models/knn.joblib'),
            'SVC': joblib.load('models/svc.joblib'),
            'Logistic Regression': joblib.load('models/lg.joblib'),
            'Gradient Boosting': joblib.load('models/gradient_boosting_classifier.joblib'),
            'Voting Hard': joblib.load('models/voting_classifier_hard.joblib'),
            'Voting Soft': joblib.load('models/voting_classifier_soft.joblib'),
            'Stacking': joblib.load('models/stacking_classifier.joblib')
        }

        
        # Charger les préprocesseurs
        label_encoders = joblib.load('processors/label_encoders.pkl')
        scaler = joblib.load('processors/scaler.pkl')
        feature_names = joblib.load('processors/feature_names.pkl')
        
        return models, label_encoders, scaler, feature_names
        
    except FileNotFoundError:
        st.error("❌ Models files are not found")
        return None, None, None, None

# Formulaire de saisie des données
st.header("📝 Loan Applicant Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal information")
    
    gender = st.selectbox("Genre", ["Male", "Female"])
    married = st.selectbox("Married", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self_Employed", ["No", "Yes"])

with col2:
    st.subheader("Financial information")
    
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=100)
    loan_amount_term = st.number_input("Loan Amount Term", min_value=0, value=360)
    credit_history = st.selectbox("Credit History", [1, 0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Bouton de prédiction
if st.button("🎯 Predict", type="primary"):
    
    # Charger les modèles
    models, label_encoders, scaler, feature_names = load_models_and_preprocessors()
    
    if models is not None:
        # Créer le dataframe d'entrée
        input_data = {
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Définir les colonnes (doivent correspondre à l'entraînement)
        categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 
                              'Self_Employed', 'Property_Area']
        numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                           'Loan_Amount_Term', 'Credit_History']
        
        # Prétraiter les données
        processed_data = preprocess_input_data(input_df, label_encoders, scaler, 
                                             numerical_columns, categorical_columns)
        
        # Réorganiser les colonnes pour correspondre à l'entraînement
        if feature_names is not None:
            processed_data = processed_data[feature_names]
        
        # Faire les prédictions avec tous les modèles
        st.header("📊 Predictions results")
        
        predictions = {}
        probabilities = {}
        
        for model_name, model in models.items():
            # Prédiction
            prediction = model.predict(processed_data)[0]
            
            # Probabilités (si disponible)
            try:
                probability = model.predict_proba(processed_data)[0]
                probabilities[model_name] = probability
            except:
                # Pour Voting Hard, calculer manuellement les "votes"
                if model_name == 'Voting Hard':
                    try:
                        # Récupérer les votes individuels des modèles
                        individual_predictions = []
                        for estimator in model.estimators_:
                            individual_pred = estimator.predict(processed_data)[0]
                            individual_predictions.append(individual_pred)
                        
                        # Calculer le pourcentage de votes pour chaque classe
                        vote_count_1 = sum(individual_predictions)
                        vote_count_0 = len(individual_predictions) - vote_count_1
                        total_votes = len(individual_predictions)
                        
                        # Créer des "pseudo-probabilités" basées sur les votes
                        prob_0 = vote_count_0 / total_votes
                        prob_1 = vote_count_1 / total_votes
                        probabilities[model_name] = [prob_0, prob_1]
                    except Exception as e:
                        st.warning(f"Could not calculate probabilities for {model_name}: {e}")
                        probabilities[model_name] = [None, None]
                else:
                    probabilities[model_name] = [None, None]
            
            predictions[model_name] = prediction
        
        # Afficher les résultats dans un tableau
        results_data = []
        for model_name, pred in predictions.items():
            prob = probabilities[model_name]
            prob_approved = prob[1] if prob[1] is not None else "N/A"
            prob_rejected = prob[0] if prob[0] is not None else "N/A"
            
            result = "✅ Approved" if pred == 1 else "❌ Refused"
            
            results_data.append({
                'Model': model_name,
                'Prediction': result,
            })
        
        # Convertir en DataFrame pour un affichage plus joli
        results_df = pd.DataFrame(results_data)
        
        # Afficher le tableau
        st.dataframe(results_df, use_container_width=True)
        
        # Résumé global
        st.subheader("🎯 Overall Summary")
        
        approved_count = sum(1 for pred in predictions.values() if pred == 1)
        total_models = len(predictions)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Models that approved", f"{approved_count}/{total_models}")
        
        with col2:
            st.metric("Models that refused", f"{total_models - approved_count}/{total_models}")
        
        with col3:
            consensus = "✅ Approved" if approved_count > total_models / 2 else "❌ Refused"
            st.metric("Consensus", consensus)
        
        # Détails des probabilités pour les modèles qui les supportent
        st.subheader("📈 Probability details")
        
        prob_models = {name: prob for name, prob in probabilities.items() 
                      if prob[0] is not None and prob[1] is not None}
        
        if prob_models:
            prob_df = pd.DataFrame({
                'Model': list(prob_models.keys()),
                'P(Refuse)': [f"{prob[0]:.2%}" for prob in prob_models.values()],
                'P(Approval)': [f"{prob[1]:.2%}" for prob in prob_models.values()]
            })
            st.dataframe(prob_df, use_container_width=True)
        
        # Explication des modèles
        st.subheader("🤖 Model explanation")
       
        model_explanations = {
            'Random Forest': "Combines multiple decision trees to produce a more robust prediction.",
            'SVM': "Finds the optimal boundary that separates the classes.",
            'KNN': "Classifies a data point based on the majority class among its nearest neighbors.",
            'Logistic Regression': "Uses a logistic function to model the probability of a binary outcome.",
            "Gradient Boosting": "Builds an ensemble of weak prediction models, typically decision trees, in a stage-wise fashion to optimize predictive performance.",
            'Voting Hard': "Uses majority voting among individual models.",
            'Voting Soft': "Averages the prediction probabilities of individual models.",
            'Stacking': "A meta-model that learns how to combine predictions from several base models."
        }


        
        for model_name, explanation in model_explanations.items():
            if model_name in predictions:
                pred_icon = "✅" if predictions[model_name] == 1 else "❌"
                st.write(f"{pred_icon} **{model_name}**: {explanation}")

# Section pour l'upload de fichier batch
st.header("📁 Batch prediction")
uploaded_file = st.file_uploader("upload a csv file to make batch prediction", 
                                type=['csv'])

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Overview of downloaded data:")
        st.dataframe(batch_data.head())
        
        if st.button("🚀Start batch predicton"):
            models, label_encoders, scaler, feature_names = load_models_and_preprocessors()
            
            if models is not None:
                # Prétraiter les données batch
                categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 
                                      'Self_Employed', 'Property_Area']
                numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                                   'Loan_Amount_Term', 'Credit_History']
                
                processed_batch_data = preprocess_input_data(batch_data, label_encoders, scaler,
                                                           numerical_columns, categorical_columns)
                
                if feature_names is not None:
                    processed_batch_data = processed_batch_data[feature_names]
                
                # Faire les prédictions
                batch_predictions = {}
                for model_name, model in models.items():
                    batch_predictions[model_name] = model.predict(processed_batch_data)
                
                # Ajouter les prédictions au dataframe original
                results_batch = batch_data.copy()
                for model_name, preds in batch_predictions.items():
                    results_batch[f'Pred_{model_name}'] = ['APRROVED' if p == 1 else 'REFUSED' for p in preds]
                
                st.success("✅ Prediction are finished!")
                st.dataframe(results_batch, use_container_width=True)
                
                # Télécharger les résultats
                csv = results_batch.to_csv(index=False)
                st.download_button(
                    label="📥 Download results",
                    data=csv,
                    file_name="predictions_loan.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        st.error(f"Error in processing data: {e}")

# Footer
st.markdown("---")
st.markdown("*Loan prediction system using Ensemble Learning*")