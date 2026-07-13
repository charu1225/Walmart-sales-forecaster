import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import pickle

print("1. Loading Walmart Dataset...")
df = pd.read_csv('Walmart.csv')

print("2. Preprocessing Data into numeric features...")
# Convert categorical variables to numeric features
df["Date"] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

#split the date into day, month, year
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

print("3. Splitting data into features (X) and Target (y)...")
# X contain s eveything the model uses to laern
X= df[['Store', 'Holiday_Flag' , 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Day', 'Month', 'Year']]
#y is teh excat values we will predict
y= df['Weekly_Sales']

#split data: 80/% to train the model and 20% to test the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("4. Training the Linear Regression model...")
# Initialize the new algorithm
model = LinearRegression()

# Train it using the exact same data
model.fit(X_train, y_train)

print("5. Evaluating the performance...")
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f" -> Mean Absolute Error (MAE): {mae}, ${mae:,.2f} # AVERAGE OFF BY AMOUNT")
print("6. Saving the trained model...")
# 'wb' means write in binary. This saves the trained brain as a file.
with open('_model.pkl', 'wb') as file:
    pickle.dump(model, file)

    print(" Finished! The trained model has been saved as '_model.pkl'. You can now use this model to make predictions on new data.")




