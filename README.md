# 🧠 Parkinson's Disease Prediction Using Machine Learning

## 📄 Overview
This project aims to develop a machine learning model to predict Parkinson's Disease (PD) using voice data. The model leverages various classification techniques, including Support Vector Machine (SVM) and XGBoost, to enhance early detection and diagnosis of PD. The system integrates data from multiple sources, preprocesses it, and applies machine learning algorithms to achieve accurate predictions.

## 📊 Data Collection
The project uses voice data for classification. Data is collected from various sources and includes:
- 🎤 Voice recordings
- 🧠 Brain MRI images
- 🧍 Posture images
- 📈 Sensor-captured data
- 📝 Handwritten notes

The data is stored in CSV format and used for model training and testing.

## 🛠 Data Preprocessing
Data preprocessing involves:
- 🔄 Handling missing values using the median for numerical data and the mode for categorical data.
- 🧹 Cleaning data by removing inaccuracies and formatting it for analysis.
- ⚖ Normalizing data for training and testing.

## 🧪 Model Evaluation
- Evaluate the models using the provided scripts or Jupyter notebooks.
- Results and metrics will be saved in the results/ directory.

## 📝 Running the Jupyter Notebooks
1. Start Jupyter Notebook by running jupyter notebook in the terminal.
2. Open the relevant notebook from the notebooks/ directory.

## 📚 Algorithms

### Support Vector Machine (SVM)
SVM is used for classification by finding the optimal hyperplane that separates different classes in the feature space.

### XGBoost
XGBoost is an advanced boosting technique that combines multiple decision trees to improve classification accuracy and model performance.

## 📈 Results
- The XGBoost model achieved an accuracy of 80-90% in detecting Parkinson’s Disease from voice data.
- Performance comparisons with the SVM model are included in the project analysis.
