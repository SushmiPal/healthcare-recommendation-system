# Healthcare Risk Assessment System - Web UI

## Overview
A professional Streamlit web application for healthcare risk assessment and personalized recommendations.

## Setup Instructions

### 1. Install Dependencies
```bashefe
pip install -r requirements.txt
```

### 2. Save Models from Notebook
Run the last cell in your notebook (cell that saves models) to generate:
- `models.pkl` - Trained Random Forest models
- `scaler_dict.pkl` - StandardScaler objects for each disease
- `oe.pkl` - OneHotEncoder for categorical features
- `X_dict_features.pkl` - Feature lists for each disease model
- `recommendation_db.pkl` - Recommendation database

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Features

### User Interface
- **Professional Design**: Modern, clean UI with gradient cards and visual indicators
- **Sidebar Input Form**: Easy-to-use form for entering patient information
- **Real-time Risk Assessment**: Instant predictions for 8 health conditions
- **Visual Risk Display**: Color-coded risk levels (High/Medium/Low)
- **Personalized Recommendations**: Content-based filtering for relevant health advice
- **Interactive Charts**: Bar charts for risk score visualization

### Health Conditions Assessed
1. Diabetes
2. Hypertension
3. High Cholesterol
4. Kidney Disease
5. Obesity
6. Low HDL
7. High LDL
8. Heart Disease

### Input Fields
- Personal Details (Age, Gender, Ethnicity, Education, Income)
- Physical Measurements (Weight, Height, BMI)
- Blood Pressure & Vital Signs
- Laboratory Values (Glucose, Cholesterol, HDL, LDL, etc.)
- Lifestyle Factors (Smoking status)
- Dietary Information (Calories, Protein, Carbs, Fat)

## Usage

1. Fill in patient information in the sidebar
2. Click "Assess Health Risks" button
3. View risk scores for all conditions
4. Review detailed risk analysis with color-coded cards
5. Read personalized recommendations for high-risk conditions
6. Explore risk score visualizations

## File Structure
```
├── app.py                    # Streamlit web application
├── save_models.py            # Helper script (reference)
├── requirements.txt          # Python dependencies
├── models.pkl               # Trained models (generated)
├── scaler_dict.pkl          # Scalers (generated)
├── oe.pkl                   # OneHotEncoder (generated)
├── X_dict_features.pkl      # Feature lists (generated)
├── recommendation_db.pkl    # Recommendations (generated)
└── healthcare system.ipynb  # Training notebook
```

## Notes

⚠️ **Important**: This system is for informational purposes only and should not replace professional medical advice. Always consult with healthcare professionals for medical decisions.

## Troubleshooting

If you encounter errors:
1. Ensure all `.pkl` files are in the same directory as `app.py`
2. Check that you've run the model saving cell in the notebook
3. Verify all dependencies are installed: `pip install -r requirements.txt`
4. Make sure you're using Python 3.8 or higher

