

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout    

# Load the dataset
try:
    data = pd.read_csv('C:/Users/ekuma/downloads/StudentsPerformance.csv')
except FileNotFoundError:
    print("Error: 'StudentsPerformance.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Preprocessing
# Define categorical and numerical features
categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
numerical_features = ['math score', 'reading score', 'writing score']

# Create a new feature 'overall_score' as the average of the three scores
data['overall_score'] = data[numerical_features].mean(axis=1)

# Define the target variable
y = data['overall_score']
X = data.drop('overall_score', axis=1)


# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply the preprocessing pipeline to the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Build the DNN model
model = Sequential([
    Dense(128, activation='relu', input_shape=[X_train.shape[1]]),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

# Train the model
history = model.fit(X_train, y_train,
                    epochs=100,
                    validation_split=0.2,
                    verbose=0) # Set verbose to 0 to suppress epoch-by-epoch output

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Model Performance on Test Data:")
print(f"Mean Absolute Error: {mae:.2f}")

# Make some predictions
predictions = model.predict(X_test[:5]).flatten()
actual_values = y_test[:5].values

#print("\nSample Predictions:")
#for i in range(len(predictions)):
    #print(f"Predicted: {predictions[i]:.2f}, Actual: {actual_values[i]:.2f}")

# Enhanced output with student names and details
# made human-readable, so both techical and non-technical users can understand
# that was done with the help of reason[]- an empty list or container that we append reasons to
# for the model's predictions, making it easier to explain why a certain score was predicted.
# This is useful for both technical and non-technical users to understand the model's predictions.
# The names are fictional and used for illustrative purposes.


names = ["Udom", "Abasienyene", "John", "Abasiama", "Itoro"]

# Print enhanced, human-readable output
for i, name in enumerate(names):
    student_index = y_test[:5].index[i]  # Get original index from y_test
    student = data.loc[student_index]  # Original row with all details

    print(f"\n{name}:")
    print(f"  Gender: {student['gender']}")
    print(f"  Parental Education: {student['parental level of education']}")
    print(f"  Lunch: {student['lunch']}")
    print(f"  Test Prep: {student['test preparation course']}")
    print(f"  Math Score: {student['math score']}")
    print(f"  Reading Score: {student['reading score']}")
    print(f"  Writing Score: {student['writing score']}")
    print(f"  Predicted Overall Score: {predictions[i]:.2f}")
    print(f"  Actual Score: {actual_values[i]:.2f}")

    # Simple logic to explain prediction
    reasons = []
    if student['test preparation course'] == 'none':
        reasons.append("no test preparation")
    if student['lunch'] == 'free/reduced':
        reasons.append("free/reduced lunch (less advantaged)")
    if student['parental level of education'] in ['some high school', 'high school']:
        reasons.append("lower parental education")

    if reasons:
        print(f"Model says: Due to {', '.join(reasons)}, I predicted slightly lower.")
    else:
        print(f"Model says: Strong background, so I expected a higher score.")