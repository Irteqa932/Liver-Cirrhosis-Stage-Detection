# Liver Cirrhosis Stage Detection Using Machine Learning

## Project Overview

Liver cirrhosis is a chronic liver disease characterized by progressive liver damage that can lead to severe health complications if not diagnosed early. Accurate staging of the disease is essential for treatment planning and patient management.

This project develops a machine learning–based predictive system that determines the stage of liver cirrhosis using patient diagnosis data, including clinical indicators and laboratory test results. The system aims to assist healthcare professionals in faster and more accurate decision-making.

---

## Objectives

- Predict liver cirrhosis stage using patient medical data
- Perform data preprocessing and exploratory analysis
- Train and compare multiple machine learning models
- Evaluate performance using standard metrics
- Identify important medical features influencing disease progression

---

## Dataset Information

- Total Records: **25,000 patients**
- Features: **19 medical attributes**
- Target Variable: **Stage (1, 2, 3)**

Features include:

- Demographics (Age, Sex)
- Clinical indicators (Ascites, Hepatomegaly, Edema)
- Laboratory tests (Bilirubin, Albumin, Platelets, Cholesterol, etc.)

---

## Technologies Used

- Python
- Google Colab
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- XGBoost
- Joblib

---

## Machine Learning Models Implemented

The following models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

---

## Results

Model performance comparison:

| Model | Accuracy |
| --- | --- |
| Logistic Regression | 60.08% |
| Decision Tree | 91.62% |
| Random Forest | 95.72% |
| XGBoost | **96.12%** |

The **XGBoost classifier** achieved the highest accuracy and was selected as the final model.

---

## Key Insights

- Ensemble models performed significantly better than linear models.
- Clinical features such as bilirubin, albumin, platelets, and prothrombin time showed strong influence on disease stage prediction.
- The model demonstrated excellent capability in identifying disease severity from patient data.

---

## Prediction System

A prediction function was developed that takes patient medical parameters as input and outputs the predicted liver cirrhosis stage.

Example Output:

```
Predicted Stage: 0
```

---

## Model Saving

The trained model and scaler were saved for deployment:

```
liver_model.pkl
scaler.pkl
```

## How to Run the Project

1. Clone the repository:

```
git clone https://github.com/your-username/liver-cirrhosis-ml.git
```

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Install dependencies:

```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

1. Run all cells.

---

## Future Improvements

- Hyperparameter tuning
- Web deployment using Streamlit
- Integration with healthcare systems
- Deep learning approaches

---
