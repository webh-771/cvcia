import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Select features and target
X = df.drop(target, axis=1)
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ========== LINEAR REGRESSION ==========
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("MAE:", mean_absolute_error(y_test, y_pred))
