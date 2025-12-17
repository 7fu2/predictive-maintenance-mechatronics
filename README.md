Project Title: Predictive Maintenance for Industrial Milling Machines

## 1. Project Overview
In the field of Mechatronics and Industry 4.0, unexpected equipment failure is a major source of cost and inefficiency. This project utilizes Machine Learning to solve this problem by implementing
a Predictive Maintenance System.  
By analyzing sensor data from industrial milling machines, I built a model capable of predicting equipment failures before they occur, allowing for proactive rather than reactive maintenance.

## 2. Problem Statement
We want to: Classify potential equipment failures for industrial milling machines.  


-Because it impacts: Production downtime, maintenance costs, and factory safety.  
-Using data from: Simulated sensor readings (Air Temperature, Process Temperature, Torque, RPM) at a per-cycle granularity.  
-Success looks like: Achieving a high accuracy (>95%) in distinguishing between healthy and failing machines.  
-Constraints: The model must handle "imbalanced data" (since failures are rare compared to normal operations).  

## 3. Data Source
I utilized the AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository. This dataset consists of 10,000 data points representing individual machine states.
Key Features (Sensors):  


-Air Temperature [K]: Ambient room temperature.  
-Process Temperature [K]: Temperature of the milling process.  
-Rotational Speed [rpm]: Speed of the spindle.  
-Torque [Nm]: Torque force applied during the process.  
-Tool Wear [min]: The accumulated usage of the cutting tool.  


Target Variable:


-Machine Failure: A binary label (0 = No Failure, 1 = Failure).  

## 4. Methodology & Tech Stack
To build this solution, I used Python due to its extensive support for data science libraries.


-Pandas: For data ingestion, cleaning, and renaming complex column headers.  
-Scikit-Learn: For splitting the data and implementing the machine learning algorithm.  
-Seaborn/Matplotlib: For visualizing the results via a Confusion Matrix.  
-Algorithm Used: Random Forest Classifier. I chose this algorithm because it is robust against noise in sensor data and provides high accuracy by averaging the results of multiple decision trees.  

## 5. Implementation Steps


My development process followed the standard Data Science lifecycle:  


-Data Preprocessing: I loaded the raw CSV data and removed irrelevant identifiers (UDI, Product ID) that do not contribute to the physical behavior of the machine.
I also renamed columns for better code readability.  
-Feature Selection: I isolated the physical sensor inputs (Temperatures, RPM, Torque, Tool Wear) as the features ($X$) and the failure status as the target ($y$).  
-Train-Test Split: I split the dataset into an 80% Training Set (to teach the model) and a 20% Testing Set (to evaluate performance on unseen data). This ensures the model isn't just memorizing the answers.  
-Model Training: I initialized a Random Forest Classifier with 100 estimators and trained it on the training data.  


## 6. The Code Implementation
Below is the full Python script used to train the model. It uses `pandas` for data processing and `RandomForestClassifier` for the prediction logic.

```python
# --- STEP 1: IMPORT LIBRARIES ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- STEP 2: LOAD AND CLEAN DATA ---
# Load the dataset
df = pd.read_csv('ai4i2020.csv')

# Rename columns to be easier to type
df.rename(columns={'Air temperature [K]': 'Air_Temp',
                   'Process temperature [K]': 'Process_Temp',
                   'Rotational speed [rpm]': 'RPM',
                   'Torque [Nm]': 'Torque',
                   'Tool wear [min]': 'Tool_Wear',
                   'Machine failure': 'Failure'}, inplace=True)

# Select the sensor data (Features) and the target (Failure)
X = df[['Air_Temp', 'Process_Temp', 'RPM', 'Torque', 'Tool_Wear']]
y = df['Failure']

# --- STEP 3: SPLIT DATA ---
# 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- STEP 4: TRAIN THE MODEL ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- STEP 5: EVALUATE ---
predictions = model.predict(X_test)

# Print Accuracy
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

```

<img width="583" height="441" alt="confusion_matrix" src="https://github.com/user-attachments/assets/1cccff8d-5c67-44e5-a0f5-8db373a91a0f" />

## 7. Results & Analysis
I evaluated the model using a test set of 2,000 unseen examples.  


Overall Accuracy: 98.35%  
-This indicates that the model correctly identified the machine's status in almost all cases.  


Confusion Matrix Breakdown:


-True Negatives (1,931): The model correctly identified healthy machines, ensuring normal operations continue without false interruptions.  
-True Positives (36): The model successfully predicted 36 failures before they happened. These represent actionable alerts that save the factory from unplanned downtime.  
-False Positives (8): Only 8 false alarms were raised. In an industrial context, this is an acceptable cost compared to the risk of a crash.  
-False Negatives (25): The model missed 25 failures. This is a known challenge with imbalanced datasets (where failures are rare), which I plan to address in future iterations.  

## 8. Future Improvements
To further enhance this project, I have identified the following next steps:


-Addressing Imbalance: Implement SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic examples of failures, helping the model learn to catch those 25 missed cases.  
-Deep Learning: Experiment with LSTMs (Long Short-Term Memory networks) to treat the data as a time-series, predicting failure based on the trend of data over time rather than a single snapshot.  
-Real-Time Dashboard: Build a simple web interface (using Streamlit) where a user can input current sensor readings and get an instant "Pass/Fail" prediction.  




Created by:  

Omar Emad Elmanhy 231023238  
Ali Mohamed Sheashae 221023201  
Zaid Walid Soliman 231023202  
