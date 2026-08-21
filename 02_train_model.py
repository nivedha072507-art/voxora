import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from voxora_core import DATA_CSV_PATH, save_model

print("Loading Voxera Sorted Dataset...")
df = pd.read_csv(DATA_CSV_PATH)

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

save_model(clf)
