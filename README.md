# Liver Cirrhosis Stage Detection Using Machine Learning

## Project Overview

```
Accuracy: 96.12% | Dataset: 25K Records | Model: XGBoost
```

Liver cirrhosis is a chronic liver disease characterized by progressive liver damage that can lead to severe health complications if not diagnosed early. Accurate staging of the disease is essential for effective treatment planning and patient management.

This project develops a machine learning–based predictive system to determine the stage of liver cirrhosis using patient diagnosis data, including laboratory test results and clinical indicators. The system aims to assist healthcare professionals in faster and more accurate decision-making.

---

## Project Highlights

*  Dataset Size: **25,000 patient records**
*  Machine Learning Models: Logistic Regression, Decision Tree, Random Forest, XGBoost
*  Best Model: **XGBoost Classifier**
*  Accuracy Achieved: **96.12%**
*  Feature Importance Analysis Performed
*  Prediction System Implemented

---

## Objectives

* Predict liver cirrhosis stages using patient clinical data
* Perform data preprocessing and exploratory analysis
* Train and compare multiple machine learning models
* Evaluate model performance using standard metrics
* Identify key medical features influencing disease progression

---

## Dataset Information

The dataset contains clinical and laboratory data of patients, including:

* Age, Sex
* Ascites, Hepatomegaly, Edema
* Bilirubin, Albumin, Cholesterol
* Platelets, Prothrombin
* Drug Type and Status
* Target Variable: **Stage (1, 2, 3)**

---

## Technologies Used

* Python
* Google Colab / Jupyter Notebook
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-learn
* XGBoost
* Joblib

---

## Machine Learning Models Implemented

The following classification algorithms were trained and compared:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* XGBoost Classifier

---

## Results

Model performance comparison:

| Model               | Accuracy   |
| ------------------- | ---------- |
| Logistic Regression | 60.08%     |
| Decision Tree       | 91.62%     |
| Random Forest       | 95.72%     |
| XGBoost             | **96.12%** |

The **XGBoost model** achieved the highest accuracy and was selected as the final model.

---

##  Key Insights

* Ensemble models significantly outperformed linear models.
* Clinical parameters such as bilirubin, albumin, platelets, and prothrombin time showed strong influence on disease severity.
* The trained model demonstrated excellent capability in predicting liver cirrhosis stages from patient data.

---

## Prediction System

A prediction function was created to input patient parameters and output the predicted liver cirrhosis stage.

Example:

```
Predicted Stage: 0
```

---

## Model Files

Due to file size limitations, trained model files are not included in the repository.

You can generate them by running the notebook. The following files will be created automatically:

```
liver_model.pkl
scaler.pkl
```

---

##  Project Structure

```
Liver-Cirrhosis-Stage-Detection/
│
├── liver_cirrhosis.csv
├── Liver_Cirrhosis_Stage_Detection.ipynb
├── README.md
└── requirements.txt
```

---

##  How to Run the Project

1️. Clone the repository:

```
git clone https://github.com/your-username/liver-cirrhosis-stage-detection.git
```

2️. Install dependencies:

```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

3️. Open the notebook in Google Colab or Jupyter Notebook.

4️. Run all cells sequentially.

---

##  Future Improvements

* Hyperparameter tuning for improved performance
* Deployment using Streamlit or Flask
* Integration with healthcare systems
* Deep learning approaches for advanced prediction
---
## Project Resources
- Google Colab Notebook: [Open Notebook](https://colab.research.google.com/drive/1V9HnnojkCVS_4wuo3QliRHFQHVt0iwjL?usp=sharing)

---
