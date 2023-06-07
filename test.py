import json
import pandas as pd
import numpy as np
import os
import pickle
import time
import subprocess
import logging
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, make_response, session
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

def test(model):
    with open('baza_test.jos') as f:
        data = json.load(f)
        
    max_skan_entries = max(len(item["skan"]) for item in data)
    columns = [f"RSSI_{i+1}" if i % 2 == 0 else f"MAC_{i+1}" for i in range(max_skan_entries * 2)]

    df = pd.DataFrame()
    for item in data:
        scan_values = []
        for scan in item["skan"]:
            scan_values.append(scan["RSSI"])
            scan_values.append(scan["MAC"])
        df = df.append(pd.Series(scan_values), ignore_index=True)

    df.columns = columns

    #df = pd.DataFrame(new_data, columns=columns)
    print(df)
    #df = pd.json_normalize(data, record_path=['skan'])
    #df = pd.concat([df.drop(columns=['XY']), pd.json_normalize(df['XY']).astype(float)], axis=1)
    enc = LabelEncoder()
    for i in range(max_skan_entries * 2):
        if not (i % 2 == 0):
            df[f'MAC_encoded_{i+1}'] = enc.fit_transform(df[f'MAC_{i+1}'])
            df.drop(columns=[f'MAC_{i+1}'], inplace=True)
    df.fillna(0, inplace=True)
    df.drop(df.columns[-2:], axis=1, inplace=True)
    print(df)
    y_pred = model.predict(df)
    file = open("Predykcje.txt", "w")
    for xy in y_pred:
        file.write(str(xy)+"\n")
    
    # Extract latitude and longitude
    latitude = [item[0] for item in y_pred]
    longitude = [item[1] for item in y_pred]

    # Create a scatter plot
    plt.plot(longitude, latitude)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Scatter Plot of Data')

    # Display the plot
    plt.show()
    
    
def trainModel():
    # Load the JSON data
    with open('baza_pozycji.json') as f:
        data = json.load(f)
        
    max_skan_entries = max(len(item["skan"]) for item in data)
    columns = [f"RSSI_{i+1}" if i % 2 == 0 else f"MAC_{i+1}" for i in range(max_skan_entries * 2)]

    df = pd.DataFrame()
    for item in data:
        xy_values = list(item["XY"].values())
        scan_values = []
        for scan in item["skan"]:
            scan_values.append(scan["RSSI"])
            scan_values.append(scan["MAC"])
        row_values = xy_values + scan_values
        df = df.append(pd.Series(row_values), ignore_index=True)


    df.drop(columns=[2], inplace=True)
    df.columns = ["X", "Y"] + columns
    df.fillna(0, inplace=True)
    print(df)
    # Convert JSON data to pandas DataFrame
    #df = pd.json_normalize(data, record_path=['skan'], meta=['XY'])
    #df = pd.concat([df.drop(columns=['XY']), pd.json_normalize(df['XY']).astype(float)], axis=1)

    # Encode the MAC column using label encoding
    enc = LabelEncoder()
    for i in range(max_skan_entries * 2):
        if not (i % 2 == 0):
            df[f'MAC_encoded_{i+1}'] = enc.fit_transform(df[f'MAC_{i+1}'].astype(str))
            df.drop(columns=[f'MAC_{i+1}'], inplace=True)
    #df['MAC_encoded'] = enc.fit_transform(df['MAC'])
    #df.drop(columns=['MAC'], inplace=True)
    
    print(df)

    # Split the data into training and testing sets
    X = df.drop(["X", "Y"], axis=1)
    y = df[['X', 'Y']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X = scaler.fit_transform(X)


    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X, y)

    # Make predictions on test data and evaluate model performance
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**.5
    print('Model Score:', score)
    #print(y_pred)
    print(rmse)
    return model
    
    
if __name__ == '__main__':
    model = trainModel()
    test(model)