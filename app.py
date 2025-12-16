"""
Healthcare Risk Assessment and Recommendation System
Professional Streamlit Web Application
"""


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import gdown

# Page configuration
st.set_page_config(
    page_title="Healthcare Recommendation System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    .risk-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
    }
    .risk-low {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
    }
    .rec-card {
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        border: none;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        cursor: pointer;
        min-height: 400px;
        display: flex;
        flex-direction: column;
    }
    .rec-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25);
    }
    .rec-card.rec-teal {
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%);
        border-left: 5px solid #06d6d0;
    }
    .rec-card.rec-purple {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        border-left: 5px solid #c084fc;
    }
    .rec-card.rec-orange {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 100%);
        border-left: 5px solid #fdba74;
    }
    .rec-card.rec-pink {
        background: linear-gradient(135deg, #be185d 0%, #ec4899 100%);
        border-left: 5px solid #f472b6;
    }
    .rec-card.rec-blue {
        background: linear-gradient(135deg, #0369a1 0%, #0ea5e9 100%);
        border-left: 5px solid #38bdf8;
    }
    .rec-card.rec-emerald {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        border-left: 5px solid #6ee7b7;
    }
    .rec-card h4 {
        color: #ffffff;
        font-size: 1.35rem;
        margin: 0 0 1.5rem 0;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    .rec-card ul {
        margin: 0;
        padding: 0;
        list-style: none;
        flex-grow: 1;
    }
    .rec-card li {
        margin: 0.9rem 0;
        color: rgba(255, 255, 255, 0.95);
        font-size: 0.95rem;
        line-height: 1.7;
        padding-left: 2rem;
        position: relative;
    }
    .rec-card li::before {
        content: "✓";
        position: absolute;
        left: 0;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 700;
        font-size: 1.1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .risk-metric {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    .risk-metric:hover {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .risk-metric-label {
        color: #cbd5e1;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .risk-metric-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    .risk-badge-high {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .risk-badge-medium {
        background: rgba(249, 115, 22, 0.2);
        color: #fdba74;
        border: 1px solid rgba(249, 115, 22, 0.4);
    }
    .risk-badge-low {
        background: rgba(34, 197, 94, 0.2);
        color: #86efac;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .recommendation-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin: 2rem 0;
    }
    @media (max-width: 768px) {
        .recommendation-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load models and preprocessors
@st.cache_resource
def load_models():
    """Load trained models and preprocessors, downloading if needed."""
    model_url = "https://drive.google.com/uc?export=download&id=1jjGY8HALMzKAG2Pli0nGYAfCV-tuIKwT"
    
    if not os.path.exists('models.pkl'):
        st.info("📥 Downloading models.pkl... This may take a few moments.")
        gdown.download(model_url, 'models.pkl', quiet=False)
        st.success("✅ Download complete!")
    
    try:
        with open('models.pkl', 'rb') as f:
            models = pickle.load(f)
        with open('scaler_dict.pkl', 'rb') as f:
            scaler_dict = pickle.load(f)
        with open('oe.pkl', 'rb') as f:
            oe = pickle.load(f)
        with open('X_dict_features.pkl', 'rb') as f:
            feature_lists = pickle.load(f)
        with open('recommendation_db.pkl', 'rb') as f:
            recommendation_db = pickle.load(f)
        return models, scaler_dict, oe, feature_lists, recommendation_db
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.stop()

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Load models
try:
    models, scaler_dict, oe, feature_lists, recommendation_db = load_models()
    st.session_state.models_loaded = True
except:
    st.error("⚠️ Please ensure all model files (models.pkl, scaler_dict.pkl, oe.pkl, X_dict_features.pkl, recommendation_db.pkl) are in the same directory as this app.")
    st.info("Run the save code in your notebook to generate these files.")
    st.stop()



def get_recommendations(patient_conditions):
    """Get recommendations based on patient conditions"""
    if not patient_conditions:
        return []
    results = []
    for condition in patient_conditions:
        if condition in recommendation_db:
            results.append((condition, recommendation_db[condition]))
    return results

def prepare_input_data(user_input, feature_list, oe):
    """Prepare user input for model prediction"""
    # Create a DataFrame with all possible features
    all_features = [
        'Age', 'Gender', 'Ethnicity', 'Education', 'Income',
        'Weight', 'Height', 'BMI', 'Pulse',
        'smoking status', 'Cigerette_last_30_days',
        'kcal', 'Protin', 'Carb', 'Fat',
        'Systolic bp', 'Diastolic bp',
        'cholesterol', 'High cholesterol', 'Low cholesterol',
        'Glucose', 'Albumin_creatinine_ratio'
    ]
    
    # Initialize DataFrame with zeros
    input_df = pd.DataFrame(0, index=[0], columns=all_features)
    
    # Fill in user inputs
    for key, value in user_input.items():
        if key in input_df.columns:
            input_df[key] = value
    
    # Calculate BMI if not provided
    if 'BMI' not in user_input or user_input['BMI'] == 0:
        if 'Weight' in user_input and 'Height' in user_input and user_input['Height'] > 0:
            input_df['BMI'] = user_input['Weight'] / ((user_input['Height'] / 100) ** 2)
    
    # One-hot encode categorical features
    categorical_cols = ['Education', 'smoking status', 'Gender', 'Income', 'Ethnicity']
    cat_data = input_df[categorical_cols]
    encoded_array = oe.transform(cat_data)
    encoded_df = pd.DataFrame(encoded_array, columns=oe.get_feature_names_out(categorical_cols))
    
    # Combine numerical and encoded features
    numerical_cols = [col for col in all_features if col not in categorical_cols]
    final_df = pd.concat([input_df[numerical_cols], encoded_df], axis=1)
    
    # Select only the features needed for this disease
    available_features = [f for f in feature_list if f in final_df.columns]
    missing_features = [f for f in feature_list if f not in final_df.columns]
    
    if missing_features:
        # Add missing features with default values
        for feat in missing_features:
            final_df[feat] = 0
    
    # Reorder columns to match feature_list
    final_df = final_df[feature_list]
    
    return final_df

def get_risk_level(risk_score):
    """Determine risk level based on score"""
    if risk_score >= 0.7:
        return "High", "🔴"
    elif risk_score >= 0.4:
        return "Medium", "🟡"
    else:
        return "Low", "🟢"

# Main App
st.markdown('<h1 class="main-header">🏥 Healthcare  Recommendation System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for user input
with st.sidebar:
    st.header("📋 Patient Information")
    st.markdown("### Personal Details")
    
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
    gender = st.selectbox("Gender", options=[1.0, 2.0], format_func=lambda x: "Male" if x == 1.0 else "Female")
    ethnicity = st.selectbox("Ethnicity", 
                            options=[1.0, 2.0, 3.0, 4.0, 6.0, 7.0],
                            format_func=lambda x: {
                                1.0: "Mexican American",
                                2.0: "Other Hispanic",
                                3.0: "Non-Hispanic White",
                                4.0: "Non-Hispanic Black",
                                6.0: "Non-Hispanic Asian",
                                7.0: "Other Race"
                            }.get(x, "Unknown"))
    education = st.selectbox("Education Level",
                            options=[1.0, 2.0, 3.0, 4.0, 5.0],
                            format_func=lambda x: {
                                1.0: "Less than 9th grade",
                                2.0: "9-11th grade",
                                3.0: "High school graduate",
                                4.0: "Some college",
                                5.0: "College graduate"
                            }.get(x, "Unknown"))
    income = st.selectbox("Income Level",
                         options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 13.0, 14.0, 15.0],
                         format_func=lambda x: f"Level {int(x)}")
    
    st.markdown("### Physical Measurements")
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
    
    # Calculate BMI automatically from weight and height
    bmi = weight / ((height / 100) ** 2)
    st.info(f"**Calculated BMI: {bmi:.2f}** (based on weight and height)")
    
    # Optional: Allow manual BMI override
    use_manual_bmi = st.checkbox("Override BMI with manual value", value=False)
    if use_manual_bmi:
        bmi = st.number_input("Enter BMI manually", min_value=10.0, max_value=60.0, value=float(bmi), step=0.1)
    
    st.markdown("### Blood Pressure & Vital Signs")
    systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=250, value=120)
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
    pulse = st.number_input("Pulse (bpm)", min_value=40, max_value=150, value=72)
    
    st.markdown("### Laboratory Values")
    glucose = st.number_input("Glucose (mg/dL)", min_value=50, max_value=500, value=100)
    cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
    hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=20, max_value=150, value=50)
    ldl = st.number_input("LDL Cholesterol (mg/dL)", min_value=50, max_value=300, value=120)
    albumin_creatinine = st.number_input("Albumin/Creatinine Ratio", min_value=0.0, max_value=500.0, value=10.0, step=0.1)
    
    st.markdown("### Lifestyle Factors")
    smoking_status = st.selectbox("Smoking Status",
                                 options=[1.0, 2.0],
                                 format_func=lambda x: "Yes" if x == 1.0 else "No")
    cigarettes = st.number_input("Cigarettes in last 30 days", min_value=0, max_value=100, value=0)
    
    st.markdown("### Dietary Information")
    kcal = st.number_input("Daily Calories (kcal)", min_value=500, max_value=5000, value=2000)
    protein = st.number_input("Daily Protein (g)", min_value=0, max_value=300, value=70)
    carbs = st.number_input("Daily Carbohydrates (g)", min_value=0, max_value=500, value=250)
    fat = st.number_input("Daily Fat (g)", min_value=0, max_value=200, value=65)
    
    predict_button = st.button("🔍 Assess Health Risks", type="primary", use_container_width=True)

# Main content area
if predict_button:
    # Prepare user input
    user_input = {
        'Age': age,
        'Gender': gender,
        'Ethnicity': ethnicity,
        'Education': education,
        'Income': income,
        'Weight': weight,
        'Height': height,
        'BMI': bmi,
        'Systolic bp': systolic_bp,
        'Diastolic bp': diastolic_bp,
        'Pulse': pulse,
        'Glucose': glucose,
        'cholesterol': cholesterol,
        'High cholesterol': hdl,
        'Low cholesterol': ldl,
        'Albumin_creatinine_ratio': albumin_creatinine,
        'smoking status': smoking_status,
        'Cigerette_last_30_days': cigarettes,
        'kcal': kcal,
        'Protin': protein,
        'Carb': carbs,
        'Fat': fat
    }
    
    # Predict risks for all diseases
    risk_scores = {}
    disease_names = {
        'diabetes': 'Diabetes',
        'hypertension': 'Hypertension',
        'cholesterol': 'High Cholesterol',
        'kidney': 'Kidney Disease',
        'obesity': 'Obesity',
        'hdl': 'Low HDL',
        'ldl': 'High LDL',
        'heart': 'Heart Disease'
    }
    
    with st.spinner("Analyzing health risks..."):
        for disease in models.keys():
            try:
                # Prepare input for this specific disease
                feature_list = feature_lists[disease]
                input_data = prepare_input_data(user_input, feature_list, oe)
                
                # Scale the input
                scaler = scaler_dict[disease]
                input_scaled = scaler.transform(input_data)
                
                # Predict
                risk_prob = models[disease].predict_proba(input_scaled)[0, 1]
                risk_scores[disease] = risk_prob
            except Exception as e:
                st.error(f"Error predicting {disease}: {str(e)}")
                risk_scores[disease] = 0.0
    
    # Display Results
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("## 📊 Health Risk Assessment Results")
    
    with col2:
        with st.expander("📋 Risk Guide", expanded=False):
            st.markdown("""
            **🔴 High Risk:** ≥ 70%  
            **🟡 Medium Risk:** 40-69%  
            **🟢 Low Risk:** < 40%
            """)
    
    st.markdown("---")
    
    # Create columns for risk scores
    cols = st.columns(4)
    col_idx = 0
    
    risk_icons = {
        'High': '⚠️',
        'Medium': '⚡',
        'Low': '✅'
    }
    
    # Sort by risk score (highest first)
    sorted_metrics = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
    
    for disease, risk_score in sorted_metrics:
        risk_level, emoji = get_risk_level(risk_score)
        badge_class = f"risk-badge-{risk_level.lower()}"
        icon = risk_icons[risk_level]
        
        with cols[col_idx % 4]:
            st.markdown(f"""
            <div class="risk-metric">
                <div class="risk-metric-label">{disease_names[disease]}</div>
                <div class="risk-metric-value">{risk_score*100:.1f}%</div>
                <span class="risk-badge {badge_class}">{icon} {risk_level} Risk</span>
            </div>
            """, unsafe_allow_html=True)
        col_idx += 1
    
    st.markdown("---")
    
    # Detailed Risk Cards
    st.markdown("## 🎯 Detailed Risk Analysis")
    
    # Sort by risk score
    sorted_risks = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
    
    for disease, risk_score in sorted_risks:
        risk_level, emoji = get_risk_level(risk_score)
        risk_class = "risk-high" if risk_score >= 0.7 else ("risk-medium" if risk_score >= 0.4 else "risk-low")
        
        st.markdown(f"""
        <div class="risk-card {risk_class}">
            <h3>{emoji} {disease_names[disease]}</h3>
            <h2>Risk Score: {risk_score*100:.1f}%</h2>
            <p>Risk Level: <strong>{risk_level}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("---")
    st.markdown("""
    <div style="margin-top: 3rem; margin-bottom: 2.5rem; padding: 0;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            <div style="width: 5px; height: 40px; background: linear-gradient(180deg, #0ea5e9 0%, #06b6d4 100%); border-radius: 3px;"></div>
            <h2 style="color: #ffffff; font-weight: 700; font-size: 2rem; margin: 0; letter-spacing: 0.5px;">
                💊 Clinical Recommendations
            </h2>
        </div>
        <p style="color: #cbd5e1; font-size: 0.95rem; margin: 0.5rem 0 0 0rem; line-height: 1.7;">
            Personalized evidence-based recommendations for each identified health condition. Click or hover for more details.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get high-risk conditions
    high_risk_conditions = [disease for disease, score in risk_scores.items() if score >= 0.4]
    
    if high_risk_conditions:
        recommendations = get_recommendations(high_risk_conditions)
        
        color_scheme = ["rec-teal", "rec-purple", "rec-orange", "rec-pink", "rec-blue", "rec-emerald"]
        disease_icons = {
            'diabetes': '🩺',
            'hypertension': '💓',
            'cholesterol': '⚕️',
            'kidney': '🫘',
            'obesity': '⚖️',
            'hdl': '💉',
            'ldl': '🔬',
            'heart': '❤️'
        }
        
        cols = st.columns(2)
        for idx, (cond_name, rec_text) in enumerate(recommendations):
            rec_items = [item.strip() for item in rec_text.strip().split('\n') if item.strip() and item.strip().startswith('•')]
            rec_html = "<ul>"
            for item in rec_items:
                clean_item = item.replace('•', '').strip()
                rec_html += f"<li>{clean_item}</li>"
            rec_html += "</ul>"
            
            color_class = color_scheme[idx % len(color_scheme)]
            icon = disease_icons.get(cond_name, '📋')
            condition_title = disease_names.get(cond_name, cond_name.title())
            
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="rec-card {color_class}">
                    <h4>{icon} {condition_title}</h4>
                    {rec_html}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("🎉 Great news! Your health risks are low. Continue maintaining a healthy lifestyle!")
        st.info("💡 General Health Tips:\n• Maintain a balanced diet\n• Exercise regularly (30 mins daily)\n• Get adequate sleep (7-9 hours)\n• Stay hydrated\n• Regular health check-ups")
    
    # Risk Summary Chart
    st.markdown("---")
    st.markdown("## 📈 Risk Score Visualization")
    
    chart_data = pd.DataFrame({
        'Disease': [disease_names[d] for d in risk_scores.keys()],
        'Risk Score (%)': [risk_scores[d] * 100 for d in risk_scores.keys()]
    })
    chart_data = chart_data.sort_values('Risk Score (%)', ascending=True)
    
    st.bar_chart(chart_data.set_index('Disease'))
    
else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>Welcome to the Healthcare  Recommendation System</h2>
        <p style="font-size: 1.2rem; color: #666;">
            Enter your health information in the sidebar and click "Assess Health Risks" to get personalized 
            risk assessments and recommendations for various health conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("👈 **Please fill in the patient information in the sidebar to begin the assessment.**")
    
    # Display system information
    with st.expander("ℹ️ About This System"):
        st.markdown("""
        ### Healthcare Recommendation System
        
        This system uses machine learning models trained on NHANES (National Health and Nutrition Examination Survey) data 
        to predict health risks for various conditions:
        
        - **Diabetes Risk**: Based on lifestyle, diet, and health metrics
        - **Hypertension Risk**: Blood pressure and related factors
        - **Cholesterol Risk**: Lipid profile assessment
        - **Kidney Disease Risk**: Kidney function indicators
        - **Obesity Risk**: Body composition analysis
        - **HDL/LDL Risk**: Cholesterol sub-type analysis
        - **Heart Disease Risk**: Comprehensive cardiovascular assessment
        
        ### How It Works
        
        1. Enter your health information in the sidebar
        2. The system analyzes your data using trained Random Forest models
        3. Receive personalized risk scores for each condition
        4. Get evidence-based recommendations for high-risk areas
        
        ### Important Note
        
        ⚠️ This system is for informational purposes only and should not replace professional medical advice. 
        Always consult with healthcare professionals for medical decisions.
        """)

