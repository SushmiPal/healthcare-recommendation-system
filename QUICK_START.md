# Quick Start Guide - Healthcare Risk Assessment UI

## Step 1: Save Models from Notebook

1. Open your `healthcare system.ipynb` notebook
2. Run all cells up to and including the model training
3. Run the **last cell** (the one that saves models) to generate:
   - `models.pkl`
   - `scaler_dict.pkl`
   - `oe.pkl`
   - `X_dict_features.pkl`
   - `recommendation_db.pkl`

## Step 2: Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn
```

Or install all at once:
```bash
pip install -r requirements.txt
```

## Step 3: Run the App

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## Step 4: Use the App

1. Fill in patient information in the sidebar
2. Click "Assess Health Risks"
3. View results and recommendations

## Troubleshooting

**Error: Model files not found**
- Make sure you've run the save models cell in the notebook
- Verify all `.pkl` files are in the same directory as `app.py`

**Error: Module not found**
- Run: `pip install streamlit pandas numpy scikit-learn`

**App won't start**
- Make sure you're in the correct directory: `cd "C:\Users\cheta\Downloads\dataset"`
- Try: `python -m streamlit run app.py`

