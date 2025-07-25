import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle

# Load data
df = pd.read_csv('canteen_data.csv')

# Encode categorical features
le_item = LabelEncoder()
le_day = LabelEncoder()
le_day.fit(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
df['Item_encoded'] = le_item.fit_transform(df['Item'])
df['Day_encoded'] = le_day.fit_transform(df['Day'])

# Features and label
X = df[['Item_encoded', 'Day_encoded']]
y = df['Quantity']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model and encoders
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(le_item, open('le_item.pkl', 'wb'))
pickle.dump(le_day, open('le_day.pkl', 'wb'))

print("Model and encoders saved.")
