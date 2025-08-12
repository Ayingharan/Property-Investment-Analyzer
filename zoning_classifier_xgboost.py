
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


prepared_data = pd.read_csv('prepared_property_dataset.csv')
raw_data = pd.read_csv('property_tax_report.csv', sep=';')
prepared_data['ZONING_CLASSIFICATION'] = raw_data['ZONING_CLASSIFICATION']


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(prepared_data['ZONING_CLASSIFICATION'].fillna("Unknown"))


X = prepared_data.drop(columns=['TAX_LEVY', 'ZONING_CLASSIFICATION'], errors='ignore')
X = X.apply(lambda col: pd.factorize(col)[0] if col.dtypes == 'object' else col)


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(y)),
    eval_metric='mlogloss',
    use_label_encoder=False,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("Zoning Classification Report using XGBoost:")
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
plt.title("XGBoost Zoning Classification - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("zoning_xgboost_confusion_matrix.png")
plt.show()


joblib.dump(model, 'zoning_xgboost_model.pkl')
joblib.dump(scaler, 'zoning_xgboost_scaler.pkl')
joblib.dump(label_encoder, 'zoning_xgboost_label_encoder.pkl')
