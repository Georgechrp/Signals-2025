import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

# --- Φόρτωση δεδομένων από προηγούμενο βήμα ---
# Προϋπόθεση: το X.npy και y.npy έχουν αποθηκευτεί (ή αλλιώς import από mfcc_extraction_and_dataset.py)
# Εδώ το κάνουμε υποθετικά inline:
from mfcc_extraction_and_dataset import X, y

# --- Επιβεβαίωση δεδομένων ---
print(f"Training set: X.shape = {X.shape}, y.shape = {y.shape}")
print(f"Ετικέτες: {Counter(y)}")

# --- 1. Least Squares (με LinearRegression) ---
print("\nΕκπαίδευση Least Squares μοντέλου...")
ls_model = LinearRegression()
ls_model.fit(X, y)

y_pred_ls = ls_model.predict(X)
y_pred_ls_binary = (y_pred_ls >= 0.5).astype(int)  # μετατροπή σε 0 ή 1
ls_acc = accuracy_score(y, y_pred_ls_binary)
print(f"Least Squares Accuracy: {ls_acc:.4f}")

# --- Αποθήκευση μοντέλου ---
joblib.dump(ls_model, "least_squares_model.pkl")


# --- 2. Πολυεπίπεδο Νευρωνικό (MLP 3 επιπέδων) ---
print("\n Εκπαίδευση MLP 3 επιπέδων...")
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=100, random_state=42)
mlp_model.fit(X, y)

# --- Αξιολόγηση στο ίδιο σύνολο ---
y_pred_mlp = mlp_model.predict(X)
mlp_acc = accuracy_score(y, y_pred_mlp)
print(f"MLP Accuracy: {mlp_acc:.4f}")

# --- Αποθήκευση MLP μοντέλου ---
joblib.dump(mlp_model, "mlp_model.pkl")

print("\nΑποθήκευση των εκπαιδευμένων μοντέλων ολοκληρώθηκε.")
