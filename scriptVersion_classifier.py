import json
import pandas as pd
import numpy as np
import os
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

def trainModel():
    # Load the JSON data
    with open('baza_pozycji.json') as f:
        data = json.load(f)
        
    max_skan_entries = max(len(item["skan"]) for item in data)
    max_entries = max_skan_entries + 1
    with open('max_entries.pkl', 'wb') as f:
        pickle.dump(max_entries, f)
    #print(max_entries)
    #columns = [f"RSSI_{i+1}" if i % 2 == 0 else f"MAC_{i+1}" for i in range(max_skan_entries * 2)]
    columns = ["RSSI_" + str(i+1) if i % 2 == 0 else "MAC_" + str(i+1) for i in range(max_skan_entries * 2)]

    df = pd.DataFrame()
    for item in data:
        xy_values = list(item["XY"].values())
        print(xy_values)
        scan_values = []
        for scan in item["skan"]:
            scan_values.append(scan["RSSI"])
            scan_values.append(scan["MAC"])
        row_values = xy_values + scan_values
        #df = df.append(pd.Series(row_values), ignore_index=True)
        df = pd.concat([df, pd.DataFrame([pd.Series(row_values)])], ignore_index=True)
    #print(df)

    #df.drop(columns=[2], inplace=True)
    df.columns = ["X", "Y", "Z"] + columns
    df.fillna(0, inplace=True)
    print(df)
    # Convert JSON data to pandas DataFrame
    #df = pd.json_normalize(data, record_path=['skan'], meta=['XY'])
    #df = pd.concat([df.drop(columns=['XY']), pd.json_normalize(df['XY']).astype(float)], axis=1)

    # Encode the MAC column using label encoding
    enc = LabelEncoder()
    for i in range(max_skan_entries * 2):
        if not (i % 2 == 0):
            #df[f'MAC_encoded_{i+1}'] = enc.fit_transform(df[f'MAC_{i+1}'].astype(str))
            df['MAC_encoded_' + str(i+1)] = enc.fit_transform(df['MAC_' + str(i+1)].astype(str))
            #df.drop(columns=[f'MAC_{i+1}'], inplace=True)
            df.drop(columns=['MAC_' + str(i+1)], inplace=True)
    #df['MAC_encoded'] = enc.fit_transform(df['MAC'])
    #df.drop(columns=['MAC'], inplace=True)
    
    print(df)

    # Split the data into training and testing sets
    X = df.drop(["X", "Y", "Z"], axis=1)
    y = df[['X', 'Y', 'Z']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X = scaler.fit_transform(X)


    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X, y)

    # Make predictions on test data and evaluate model performance
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**.5
    print('Model Score:', score)
    #print(y_pred)
    print(rmse)
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
        

def loadModel():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def test(model, dataPath):
    if dataPath:
        with open(dataPath) as f:
            data = json.load(f)
    else: raise Exception("Nie podano pliku")
    
    with open('max_entries.pkl', 'rb') as f:
        max_entries = pickle.load(f)
        
    #max_skan_entries = max(len(item["skan"]) for item in data)
    max_skan_entries = max_entries
    max_columns = max_skan_entries * 2
    #columns = [f"RSSI_{i+1}" if i % 2 == 0 else f"MAC_{i+1}" for i in range(max_skan_entries * 2)]
    columns = ["RSSI_" + str(i+1) if i % 2 == 0 else "MAC_" + str(i+1) for i in range(max_skan_entries * 2)]

    df = pd.DataFrame()
    for item in data:
        scan_values = []
        for scan in item["skan"]:
            scan_values.append(scan["RSSI"])
            scan_values.append(scan["MAC"])
        df = pd.concat([df, pd.DataFrame([pd.Series(scan_values)])], ignore_index=True)
        #df = df.append(pd.Series(scan_values), ignore_index=True)

    df_columns = df.shape[1]
    if df_columns < max_columns:
        missing_columns = max_columns - df_columns
        #missing_data = pd.DataFrame(np.zeros((df.shape[0], missing_columns)), columns=[f'Missing_{i+1}' for i in range(missing_columns)])
        missing_data = pd.DataFrame(np.zeros((df.shape[0], missing_columns)), columns=['Missing_' + str(i+1) for i in range(missing_columns)])
        df = pd.concat([df, missing_data], axis=1)

    df.columns = columns

    #df = pd.DataFrame(new_data, columns=columns)
    #print(df)
    #df = pd.json_normalize(data, record_path=['skan'])
    #df = pd.concat([df.drop(columns=['XY']), pd.json_normalize(df['XY']).astype(float)], axis=1)
    enc = LabelEncoder()
    for i in range(max_skan_entries * 2):
        if not (i % 2 == 0):
            #df[f'MAC_encoded_{i+1}'] = enc.fit_transform(df[f'MAC_{i+1}'])
            df['MAC_encoded_' + str(i+1)] = enc.fit_transform(df['MAC_' + str(i+1)].astype(str))
            #df.drop(columns=[f'MAC_{i+1}'], inplace=True)
            df.drop(columns=['MAC_' + str(i+1)], inplace=True)
    df.fillna(0, inplace=True)
    df.drop(df.columns[-2:], axis=1, inplace=True)
    #print(df)
    y_pred = model.predict(df)
    predictions = y_pred.tolist()
    data = {'XY': predictions}
    return data

def main(argv):
    if not os.path.exists('./model.pkl'):
        trainModel()
    #argv = "D:/MachineLearning/baza_test.jos"
    #print(argv)
    model = loadModel()
    with open("./predictions.json", "w") as file:
        json.dump(test(model, argv[1]),file)
    #return(test(model, argv[1]))

if __name__ == "__main__":
    main(sys.argv)