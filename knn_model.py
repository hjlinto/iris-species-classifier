import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Load Iris dataset from CSV file
df = pd.read_csv('Iris.csv')

# Display first five rows to verify loading
print("First 5 rows of the dataset:\n", df.head(5))

# Declare independent and dependent variables
dependent_variables = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
independent_variable = df['Species']

# Define model with respect to 5 most similar data points
knn_model = KNeighborsClassifier(n_neighbors=3)