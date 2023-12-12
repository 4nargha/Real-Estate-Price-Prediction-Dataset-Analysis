import pickle 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from flask import Flask,request,jsonify,render_template


application=Flask(__name__)
app=application


Reg = pickle.load(open(r'C:\Users\Owner\ML_project\Housingdata_project\notebooks\linear_model.pkl', 'rb'))
best_model_rf = pickle.load(open(r'C:\Users\Owner\ML_project\Housingdata_project\notebooks\random_forest_model.pkl', 'rb'))
scaler = pickle.load(open(r'C:\Users\Owner\ML_project\Housingdata_project\notebooks\scaler.pkl', 'rb'))


data = pd.read_csv("C:\\Users\\Owner\\ML_project\\Housingdata_project\\Dataset\\data.csv")
# Extract the unique city values from the 'city' column
available_cities = data['city'].unique().tolist()
# Extract the one-hot encoding columns from the loaded data
city_columns = [col for col in data.columns if col.startswith('city_')]

@app.route('/')
def index():
    return render_template('index.html', available_cities=available_cities)

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        # Extracting input features from the form
        bedrooms = float(request.form.get('bedrooms'))
        bathrooms = float(request.form.get('bathrooms'))
        sqft_living = float(request.form.get('sqft_living'))
        sqft_lot = float(request.form.get('sqft_lot'))
        floors = float(request.form.get('floors'))
        condition = float(request.form.get('condition'))
        sqft_basement = float(request.form.get('sqft_basement'))
        yr_built = float(request.form.get('yr_built'))
        yr_renovated = float(request.form.get('yr_renovated'))
        city = request.form.get('city') 

        # Create a DataFrame with the input features
        input_data = pd.DataFrame({
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'sqft_living': [sqft_living],
            'sqft_lot': [sqft_lot],
            'floors': [floors],
            'condition': [condition],
            'sqft_basement': [sqft_basement],
            'yr_built': [yr_built],
            'yr_renovated': [yr_renovated],
            'city': [city] 
        })

         # Check if all the necessary columns are present in input_data
        required_columns = ['date', 'price', 'waterfront', 'view', 'sqft_above', 'street', 'city', 'statezip', 'country']
        for col in required_columns:
            if col not in input_data.columns:
                raise ValueError(f"'{col}' column not found in input_data.")


      

        # Check if 'city' is present in the original DataFrame
        if 'city' not in data.columns:
            raise ValueError("'city' column not found in the original DataFrame.")

        # One-hot encode the 'city' column using the dynamically obtained list of available cities
        data_encoded = pd.get_dummies(input_data, columns=['city'], prefix='city', dummy_na=False)

        # Ensure that the input_data_encoded DataFrame has all the city columns
        for col in city_columns:
            if col not in data_encoded.columns:
                data_encoded[col] = 0

        # Reorder the columns to match the order during training
        data_encoded = data_encoded[data.columns]
        
        

        # Scale the data using the pre-trained scaler
        new_data_sc = scaler.transform(data_encoded)

        # Make predictions using the pre-trained RandomForestRegressor
        result = best_model_rf.predict(new_data_sc)

        return render_template('index.html', result=result[0])

    except Exception as e:
        # Log the exception
        app.logger.error(f'An error occurred: {str(e)}')
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)