import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load the dataset
data = pd.read_csv(r"C:\ML Model Deployment\Salary Data.csv")
data.dropna(inplace=True)
# Define features and target
numeric_features = ['Age', 'Years of Experience']
categorical_features = ['Gender', 'Education Level', 'Job Title']

# Preprocessing for numerical features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first'))
])

# Bundle preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Append regressor to preprocessing pipeline
regressor = Pipeline(steps=[('preprocessor', preprocessor),
                             ('regressor', RandomForestRegressor(n_estimators=10))])

# Split the data into features and target
X = data.drop('Salary', axis=1)  # Keep all features, including those with missing values
y = data['Salary']

# Train the machine learning model
regressor.fit(X, y)

# Save the trained model to disk
pickle.dump(regressor, open("model.pkl", "wb"))
