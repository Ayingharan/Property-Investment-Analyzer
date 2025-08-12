import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.callbacks import Callback
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


class StopOnHighAccuracy(Callback):
    def __init__(self, threshold=0.999):
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("valid_accuracy")
        if val_acc and val_acc >= self.threshold:
            print(f" Early stopping: Accuracy reached {val_acc:.4f} on epoch {epoch+1}")
            raise KeyboardInterrupt


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


tabnet_model = TabNetClassifier()
try:
    tabnet_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        eval_metric=['accuracy'],
        max_epochs=100,
        patience=10,
        batch_size=256,
        virtual_batch_size=128,
        callbacks=[StopOnHighAccuracy(threshold=0.999)]
    )
except KeyboardInterrupt:
    print(" Training interrupted due to reaching 0.999 accuracy.")


y_pred = tabnet_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\nZoning Classification Report using TabNet:")
print(f"Overall Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("TabNet Zoning Classification - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("zoning_tabnet_confusion_matrix.png")
plt.show()


joblib.dump(tabnet_model, 'zoning_tabnet_model.pkl')
joblib.dump(scaler, 'zoning_tabnet_scaler.pkl')
joblib.dump(label_encoder, 'zoning_tabnet_label_encoder.pkl')
