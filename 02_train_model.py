import hashlib
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report

from model_loader import CHECKSUM_FILE

MODEL_FILE = 'voxera_7signs_model.pkl'

print("Loading Voxera Sorted Dataset...")
df = pd.read_csv('voxera_7signs_data.csv')

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training ExtraTrees Model...")
clf = ExtraTreesClassifier(
    n_estimators=250, 
    max_depth=18, 
    min_samples_split=2, 
    random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

with open(MODEL_FILE, 'wb') as f:
    pickle.dump(clf, f)

with open(MODEL_FILE, 'rb') as f:
    digest = hashlib.sha256(f.read()).hexdigest()

# Record the digest so the inference scripts will accept this model.
with open(CHECKSUM_FILE, 'w', encoding='utf-8') as f:
    f.write(
        "# SHA-256 digests of the model files VOXORA is allowed to unpickle.\n"
        f"# Regenerate with: sha256sum {MODEL_FILE}\n"
        f"{digest}  {MODEL_FILE}\n"
    )

print(f"Saved '{MODEL_FILE}' (sha256 {digest})")
