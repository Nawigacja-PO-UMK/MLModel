import json
import pandas as pd
import numpy as np
import os
import pickle
import time
from flask import Flask, request, jsonify, make_response, session
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

def trainModel():
    # Load the JSON data
    with open('baza_pozycji.json') as f:
        data = json.load(f)

    # Convert JSON data to pandas DataFrame
    df = pd.json_normalize(data, record_path=['skan'], meta=['XY'])
    df = pd.concat([df.drop(columns=['XY']), pd.json_normalize(df['XY']).astype(float)], axis=1)

    # rows = []
    # for entry in data:
    #     xy = entry['XY']
    #     skan = entry['skan']
    #     row = {
    #         'X': xy['X'],
    #         'Y': xy['Y'],
    #         'Z': xy['Z']
    #     }
    #     for i, access_point in enumerate(skan):
    #         row[f'MAC_{i+1}'] = access_point['MAC']
    #         row[f'RSSI_{i+1}'] = access_point['RSSI']
    #     rows.append(row)

    # df = pd.DataFrame(rows)


    # Encode the MAC column using one-hot encoding
    #enc = OneHotEncoder()
    #MAC_encoded = enc.fit_transform(df[['MAC']])
    #MAC_encoded = pd.DataFrame(MAC_encoded.toarray(), columns=enc.get_feature_names_out(['MAC']))
    #df = pd.concat([df.drop('MAC', axis=1), MAC_encoded], axis=1)

    # Encode the MAC column using label encoding
    enc = LabelEncoder()
    df['MAC_encoded'] = enc.fit_transform(df['MAC'])
    df.drop(columns=['MAC'], inplace=True)

    # Split the data into training and testing sets
    X = df[['MAC_encoded', 'RSSI']]
    y = df[['X', 'Y']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the neural network model
    #model = MLPRegressor(hidden_layer_sizes=(20, 10), max_iter=10000, activation='relu', solver='adam', alpha=0.0001, 
    #                     batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
    #                     momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, 
    #                     beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000, 
    #                     verbose=False, warm_start=False)
    #model.out_activation_ = 'identity'


    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

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



def get_current_timestamp():
    # Return the current timestamp
    return int(time.time())


def store_data_with_timestamp(data, timestamp):
    # Create a dictionary with data and timestamp
    entry = {
        'data': data,
        'timestamp': timestamp
    }

    # Load existing data from the database (if any)
    existing_data = []
    if os.path.exists('database.json'):
        with open('database.json', 'r') as f:
            existing_data = json.load(f)

    # Add the new entry to the existing data
    existing_data.append(entry)

    # Write the updated data back to the database
    with open('database.json', 'w') as f:
        json.dump(existing_data, f)
        
        
def retrieve_data_with_timestamps(x):
    existing_data = []
    if os.path.exists('database.json'):
        with open('database.json', 'r') as f:
            existing_data = json.load(f)

    current_timestamp = get_current_timestamp()
    threshold_timestamp = current_timestamp - x

    data = []
    for entry in existing_data[::-1]:  # Reverse order to start from the latest entry
        timestamp = entry['timestamp']
        if timestamp < threshold_timestamp:
            break  # Break the loop when reaching older data
        data.append(entry['data'])

    return data


def create_vector(data):
    # Preprocess and transform the data into a format compatible with the model's input
    vector = []
    
    df = pd.json_normalize(data, record_path=['skan'], meta=['XY'])
    df = pd.concat([df.drop(columns=['XY']), pd.json_normalize(df['XY']).astype(float)], axis=1)
    
    enc = LabelEncoder()
    df['MAC_encoded'] = enc.fit_transform(data['MAC'])
    df.drop(columns=['MAC'], inplace=True)

    for entry in df:
        # Assuming each entry in data is a dictionary with 'X', 'Y', 'MAC_encoded', and 'RSSI' keys
        mac_encoded = entry['MAC_encoded']
        rssi = entry['RSSI']

        # Create a feature vector for the entry
        entry_vector = [mac_encoded, rssi]

        vector.append(entry_vector)

    return vector



app = Flask(__name__)
model = None
app.secret_key = os.urandom(24)

@app.route('/data', methods=['POST'])
def receive_data():
    session_id = session.get('session_id')
    data = request.json  # Assuming data is sent in JSON format
    timestamp = get_current_timestamp()  # Implement your timestamp generation logic
    # Store data and timestamp in the database
    store_data_with_timestamp(data, timestamp)
    return 'Data received successfully'

@app.route('/predictions', methods=['GET'])
def get_predictions():
    session_id = session.get('session_id')
    # Retrieve the last x timestamps and corresponding data from the database
    data = retrieve_data_with_timestamps(3)
    vector = create_vector(data)  # Implement your vector creation logic
    predictions = model.predict(vector)  # Pass vector to your machine learning model
    return jsonify(predictions)

@app.route('/start_session', methods=['POST'])
def start_session():
    if 'session_id' not in session:
        session_id = os.urandom(24).hex()
        session['session_id'] = session_id
        return make_response('Session started')
    else:
        return make_response('Session already started')

if __name__ == '__main__':
    if not os.path.exists('./model.pkl'):
        trainModel()
    model = loadModel()
    
    app.run()