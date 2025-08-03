# 🚗 Car Evaluation Using Random Forest

This Streamlit app uses a Random Forest Classifier to predict the **car evaluation class** based on several input attributes.

## 📊 Dataset Columns
- Buying
- Maintenance
- Number of Doors
- Persons Capacity
- Luggage Boot Size
- Safety
- Target: `class` (unacc, acc, good, vgood)

## 🔧 Tech Stack
- Python, Pandas, Scikit-learn
- Streamlit (Web App UI)

## 🚀 To Run Locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
