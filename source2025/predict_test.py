import librosa
import numpy as np
import joblib
import csv

# --- Ρυθμίσεις ---
test_file = "S01_U04.CH4.wav"               # Το test αρχείο
model_path = "mlp_model.pkl"         # Μοντέλο προς χρήση (μπορείς να αλλάξεις σε least_squares_model.pkl)
output_csv = "results.csv"           # Τελικό αρχείο με προβλέψεις
sr = 16000                           # Sample rate
frame_duration = 0.025               # 25ms
hop_duration = 0.010                 # 10ms

# --- Υπολογισμός παραμέτρων σε samples ---
frame_length = int(frame_duration * sr)
hop_length = int(hop_duration * sr)

# --- 1. Φόρτωση του test.wav και εξαγωγή MFCCs ---
print("Φόρτωση test.wav και εξαγωγή MFCC...")
y, sr = librosa.load(test_file, sr=sr)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)
mfcc = mfcc.T  # (n_frames, 13)
print(f"MFCC shape: {mfcc.shape}")

# --- 2. Φόρτωση εκπαιδευμένου μοντέλου ---
print("Φόρτωση εκπαιδευμένου μοντέλου...")
model = joblib.load(model_path)

# --- 3. Πρόβλεψη για κάθε frame ---
print("Πρόβλεψη ανά frame...")
predictions = model.predict(mfcc)

# --- 4. Ομαδοποίηση συνεχόμενων frames με ίδια πρόβλεψη ---
print("Ομαδοποίηση προβλέψεων σε χρονικά διαστήματα...")
segments = []
current_label = predictions[0]
start_frame = 0

for i in range(1, len(predictions)):
    if predictions[i] != current_label:
        start_time = start_frame * hop_duration
        end_time = i * hop_duration
        label_text = "foreground" if current_label == 1 else "background"
        segments.append(["test.wav", round(start_time, 2), round(end_time, 2), label_text])
        start_frame = i
        current_label = predictions[i]

# Προσθήκη τελευταίου τμήματος
end_time = len(predictions) * hop_duration
label_text = "foreground" if current_label == 1 else "background"
segments.append(["test.wav", round(start_frame * hop_duration, 2), round(end_time, 2), label_text])

# --- 5. Αποθήκευση αποτελέσματος σε .csv ---
print(f"Αποθήκευση στο αρχείο {output_csv}...")
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Audiofile", "start", "end", "class"])
    writer.writerows(segments)

print("Ολοκληρώθηκε η πρόβλεψη και αποθήκευση.")
