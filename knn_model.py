import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load Iris dataset from CSV file
df = pd.read_csv('Iris.csv')

# Declare independent and dependent variables
dependent_variables = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
independent_variable = df['Species']

# Split dataset into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(dependent_variables, independent_variable, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define model with respect to 5 most similar data points
knn_model = KNeighborsClassifier(n_neighbors=3)

# Train the KNN model
knn_model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = knn_model.predict(X_test_scaled)

# Evaluate the KNN model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Print evaluation metrics
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1: {f1:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-Validation
X_scaled = scaler.fit_transform(dependent_variables)
cv_scores = cross_val_score(knn_model, X_scaled, independent_variable, cv=5, scoring='accuracy')
formatted_scores = [f"{score:.2f}" for score in cv_scores]

# Display Cross-Validation results
print("\nCross-Validation Scores:", formatted_scores)
print(f"Mean Accuracy: {cv_scores.mean():.2f}")
print(f"Standard Deviation: {cv_scores.std():.2f}")

# Hyperparameter Tuning
param_grid = {'n_neighbors': list(range(1, 20))}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_scaled, independent_variable)

# Display Hyperparameter Tuning results
print("\nGrid Search Results:")
print(f"Best k (n_neighbors): {grid_search.best_params_['n_neighbors']}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.2f}")