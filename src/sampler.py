import pandas as pd
import time
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:2727/api/v1/measurements")

def send(measure):
    resp = requests.post(API_URL, json=measure)
    resp.raise_for_status()

if __name__ == '__main__':
    bpm = pd.read_csv('data/bpm.csv', index_col=0)
    uterus = pd.read_csv('data/uterus.csv', index_col=0)
    bpm_i = uterus_i = 0
    prev_time = 0
    patient_id = "0"

    while True:
        # логика как у тебя, но вместо add_measurement делаем POST
        if uterus_i < uterus.shape[0] and bpm_i < bpm.shape[0] and bpm.iloc[bpm_i].time_sec >= uterus.iloc[uterus_i].time_sec:
            m = {
                "patient_id": patient_id,
                "timestamp": float(uterus.iloc[uterus_i].time_sec),
                "type": "uterus",
                "value": float(uterus.iloc[uterus_i].value)
            }
            send(m)
            time.sleep((uterus.iloc[uterus_i].time_sec - prev_time) / 10)
            prev_time = uterus.iloc[uterus_i].time_sec
            uterus_i += 1
        elif bpm_i < bpm.shape[0]:
            m = {
                "patient_id": patient_id,
                "timestamp": float(bpm.iloc[bpm_i].time_sec),
                "type": "bpm",
                "value": float(bpm.iloc[bpm_i].value)
            }
            send(m)
            time.sleep((bpm.iloc[bpm_i].time_sec - prev_time) / 10)
            prev_time = bpm.iloc[bpm_i].time_sec
            bpm_i += 1
        else:
            break
