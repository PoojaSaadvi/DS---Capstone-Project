# DS---Capstone-Project
Project 1 AnomaData (Automated Anomaly Detection for Predictive Maintenance)

# AnomaData: Predictive Maintenance Model

## Description:
This project aims to develop a predictive maintenance model using machine learning techniques to detect anomalies in machinery and predict potential failures before they occur. The goal is to assist industries in preventing costly equipment breakdowns by detecting anomalies early.

## Prerequisites:
- Python 3.x
- Jupyter Lab or Jupyter Notebook

## Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/AnomaData-Predictive-Maintenance.git
   cd AnomaData-Predictive-Maintenance
2. Install the required Python libraries using pip:
   `pip install pandas scikit-learn matplotlib`
   
## 1. Training the Model:
- Open the provided Jupyter notebook file (`Cape stone Project 1 AnomaData (Automated Anomaly Detection for Predictive Maintenance) (1).ipynb`) in Jupyter Lab or Jupyter Notebook.
- Run the cells sequentially to:
  - Load and preprocess the dataset.
  - Train the model using the Random Forest Classifier.
  - Save the trained model as a `.pkl` file.

## 2. Making Predictions:
Once the model is trained and saved as a `.pkl` file:

To load the model and make predictions, use the following code in a separate Jupyter notebook cell or Python script:

```python
import pickle
import pandas as pd

# Load the trained model
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the new test data (ensure it has the same features as the training data)
new_data = pd.read_csv('new_test_data.csv')

# Make predictions
predictions = model.predict(new_data)
print(predictions)
```
## 3. Model Evaluation:
The model was evaluated using the following metrics:
- **Accuracy**: 85% on the test set.
- **Precision, Recall, F1-score**: An F1-score of 0.82.
- **Confusion Matrix**: A high number of true positives indicates effective anomaly detection.

## Future Work:
- **Experiment with Other Algorithms**: Try algorithms like XGBoost, Gradient Boosting, or neural networks for potentially improved performance.
- **Data Augmentation**: More data or simulated anomalies could improve model accuracy.
- **Advanced Anomaly Detection**: Test models such as Isolation Forest or One-Class SVM.
- **Time-Series Models**: Consider models like LSTM or RNN for better handling of sequential data.
- **Real-Time Deployment**: Work on real-time anomaly detection for industrial use cases.

