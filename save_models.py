"""
Script to save trained models, scalers, and preprocessors for the Streamlit app.
Run this after training your models in the notebook.
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# This script should be run after all models are trained in the notebook
# It will save the necessary components for the Streamlit app

print("This script should be run from the notebook after training.")
print("Please run the following code in a notebook cell to save the models:")
print("""
import pickle

# Save models, scalers, and preprocessors
with open('models.pkl', 'wb') as f:
    pickle.dump(models, f)

with open('scaler_dict.pkl', 'wb') as f:
    pickle.dump(scaler_dict, f)

with open('oe.pkl', 'wb') as f:
    pickle.dump(oe, f)

with open('X_dict_features.pkl', 'wb') as f:
    # Save the feature lists for each disease
    feature_lists = {disease: list(X_dict[disease].columns) for disease in X_dict.keys()}
    pickle.dump(feature_lists, f)

with open('recommendation_db.pkl', 'wb') as f:
    pickle.dump(recommendation_db, f)

print("Models and preprocessors saved successfully!")
""")


