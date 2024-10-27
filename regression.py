import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load your dataset
data = pd.read_csv('C:/Users/raina/Development/Data Glacier/Deployment Flask/heights.csv')

# Split dataset into features and target variable
X = data[['mheight']]  # Mother's height
y = data['dheight']    # Daughter's height

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a .pkl file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
