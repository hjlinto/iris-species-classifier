import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load Iris dataset from CSV file
df = pd.read_csv('Iris.csv')

# Display first five rows to verify loading
print("First 5 rows of the dataset:\n", df.head(5))

# Declare independent and dependent variables
dependent_variables = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
independent_variable = df['Species']

# Split dataset into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(independent_variable, independent_variable, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define model with respect to 5 most similar data points
knn_model = KNeighborsClassifier(n_neighbors=3)

# Train the KNN model
knn_model.fit(X_train_scaled, y_train)