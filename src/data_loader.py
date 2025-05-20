import pandas as pd
import numpy as np
import io

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            return data
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    return None

def load_metadata(uploaded_file):
    if uploaded_file is not None:
        try:
            metadata = pd.read_csv(uploaded_file)
            return metadata
        except Exception as e:
            raise Exception(f"Error loading metadata: {str(e)}")
    return None

def save_example_data():
    data = """age,Sex,Pneumonia,PH,DiaPr,Respiratory rate,SPO2,GCS,SysPr,Pulse rate,SM PY,smoker,ex sm years,hospitalizations,(MT)
70,1,0,7.22,96,35,31,15,136,100,50,2,,,0
73,1,0,7.4,90,90,92,15,140,20,100,1,,,0
32,0,0,7.45,80,30,93,15,120,115,5,2,,,2
69,1,1,7.47,76,30,85,15,134,85,60,2,,,0
70,1,1,7.4,90,20,85,,180,74,60,2,,,0
52,0,0,7.42,86,28,92,,163,99,50,2,,,0
57,1,2,7.42,72,12,97,15,134,73,,3,,,2
84,0,0,7.33,,,,,,,,0,,,0
27,0,0,7.47,65,18,94,15,103,18,,4,,,0
70,0,0,7.47,,,,15,,120,,3,,,0"""
    
    with open("assets/example_data.csv", "w") as f:
        f.write(data)