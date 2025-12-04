#!/usr/bin/env python
# coding: utf-8

# In[3]:


# hardware implementation
import time
import serial
import numpy as np

class MCCAIDevice:
    def _init_(self, port_sampling, port_read, sampling_period_ms, read_period_ms):
        # Configure serial ports (example parameters)
        self.sampling = serial.Serial(port_sampling, baudrate=115200, timeout=sampling_period_ms / 1000.0)
        self.read = serial.Serial(port_read, baudrate=115200, timeout=read_period_ms / 1000.0)

        self.device_name = ""
        self.device_number = 0
        self.data_type = np.float64

class NIDAQAIDevice:
    def _init_(self, port_sampling,port_read, sampling_period_ms, read_period_ms):
        self.sampling = serial.Serial(port_sampling, baudrate=115200, timeout=sampling_period_ms / 1000.0)
        self.read = serial.Serial(port_read, baudrate=115200, timeout=read_period_ms / 1000.0)
        self.device_name = Dev2
        self.data_type = np.float64
        self.pk2pk_volts = 5
        self.offset_volts = 2.5




# In[6]:


#conncetion of pump to the organ
import numpy as np

class NIDAQDCDevice:
    def __init__(self, device_name="Dev2", line="ai1", flow_range=(0, 50), cal_pt1_volts=0.1, cal_pt1_flow=0.868, cal_pt2_volts=5, cal_pt2_flow=49.23):
        self.device_name = device_name
        self.line = line
        self.flow_range = flow_range
        self.cal_pt1_volts = cal_pt1_volts
        self.cal_pt1_flow = cal_pt1_flow
        self.cal_pt2_volts = cal_pt2_volts
        self.cal_pt2_flow = cal_pt2_flow
        # Calculate calibration slope and offset
        self.slope = (self.cal_pt2_flow - self.cal_pt1_flow) / (self.cal_pt2_volts - self.cal_pt1_volts)
        self.offset = self.cal_pt1_flow - self.slope * self.cal_pt1_volts

    def read_voltage(self):
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(f"{self.device_name}/{self.line}")
            voltage = task.read()
        return voltage

    def voltage_to_flow(self, voltage):
        return self.slope * voltage + self.offset

    def get_flow(self):
        flow = self.voltage_to_flow(voltage)
        # Clamp to flow range
        flow = np.clip(flow, self.flow_range[0], self.flow_range[1])
        return flow




# In[20]:


import os
print(os.getcwd())
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Read your dataset(s)

df1 = pd.read_csv ("C:/Users/NAVYA/Downloads/kidney_transplant.csv")
df2 = pd.read_csv ("C:/Users/NAVYA/Downloads/high_risk.csv")
merged_df = pd.merge(df1, df2, on="Patient_ID")
df = merged_df

# 2. One‑hot encode ONLY the categorical columns
categorical_cols = [
    "Patient_Blood",
    "Organ_Required",
    "Diagnosis_Result",
    "Biological_Markers",
    "Organ_Status",
    "Donor_Blood",
    "Organ_Donated",
    "Donor_Medical_Approval",
    "Match_Status",
    "Organ_Condition",
    "Organ_Tracking"
]

df = pd.get_dummies(df, columns=categorical_cols)

# 3. Define the target column (single column, not 5 dummy actions)
# Example: suppose you have a column "action"
y = df["Organ_Status_Matched"]                  # classification label
X = df.drop(columns=[col for col in df.columns if 'Organ_Status_'in col])   # all other features

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# 7. Example prediction (must match X columns order and count)
# Build a DataFrame with same columns as X instead of raw np.array
sample_data = ({

    "Patient_Blood_O": 1,
    "Patient_Blood_A": 0,
    "Patient_Blood_B": 0,
    "Patient_Blood_AB": 0,  
    "Organ_Required_Kidney": 1,
    "Diagnosis_Result_Positive": 0,
    "Diagnosis_Result_Negative": 1,
    "Biological_Markers_High": 1,
    "Donor_Blood_O": 1,
    "Donor_Blood_A": 0,
    "Donor_Blood_B": 0, 
    "Donor_Blood_AB": 0,
    "Organ_Donated_Yes": 1,
    "Donor_Medical_Approval_Approved": 1,
    "Match_Status_Good": 1,
    "Organ_Condition_Excellent": 0,
    "Organ_Condition_Good": 1,
    "Organ_Tracking_OnTime": 0,
    "Organ_Tracking_Delayed": 1,

})
for col in X.columns:
    if col not in sample_data:
        sample_data[col] = 0
new_input = pd.DataFrame([sample_data])  # Single row as list
prediction = model.predict(new_input)
print("Predicted action:", prediction[0])


# In[ ]:




