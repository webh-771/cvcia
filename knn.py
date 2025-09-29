import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('your_file.csv')  # Change to actual filename

# ========== EDA & PREPROCESSING ==========
# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)
for col in df.select_dtypes(include=['object']):
    df[col].fillna(df[col].mode()[0], inplace=True)
# Remove duplicates
df.drop_duplicates(inplace=True)

# Encode categorical columns except target
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
target = 'target_column'  # Change to actual target name
if target in categorical_columns:
    categorical_columns.remove(target)

le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Encode target if categorical
if df[target].dtype == 'object' or df[target].nunique() < 20:
    df[target] = le.fit_transform(df[target])

# Select features and target
X = df.drop(target, axis=1)
y = df[target]

# Scale features for KNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# ========== KNN CLASSIFICATION ==========
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
