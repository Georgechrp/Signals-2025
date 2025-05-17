import os
import librosa
import json
import numpy as np
from collections import Counter

# ----- Ρυθμίσεις ------
speech_folder = 'C:/Users/georg/Downloads/train/speech'
noise_folder = 'C:/Users/georg/Downloads/train/noise'
output_folder = 'output_json'
max_files = 20  # Όριο αρχείων για να περιορίσουμε την επεξεργασία

# --- Δημιουργία φακέλων ---
os.makedirs(f"{output_folder}/speech", exist_ok=True)
os.makedirs(f"{output_folder}/noise", exist_ok=True)

# --- Συνάρτηση εξαγωγής MFCC ---
def extract_mfcc(file_path, sr=16000, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                    n_fft=frame_length, hop_length=hop_length)
        return mfcc.T  # (frames, 13)
    except Exception as e:
        print(f"Σφάλμα με {file_path}: {e}")
        return None

# --- Βοηθητική συνάρτηση αποθήκευσης MFCC σε JSON ---
def process_folder(input_folder, output_subfolder, label_name):
    count = 0
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                print(f"Ανάλυση ({label_name}): {file}")
                mfcc = extract_mfcc(full_path)
                if mfcc is not None:
                    mfcc_list = mfcc.tolist()
                    out_filename = os.path.splitext(file)[0] + ".json"
                    out_path = os.path.join(output_folder, output_subfolder, out_filename)
                    with open(out_path, "w") as f:
                        json.dump(mfcc_list, f)
                    print(f" Αποθηκεύτηκε: {out_path}")
                    count += 1
                if count >= max_files:
                    return

# --- 1. Αποθήκευση JSONs από speech και noise ---
process_folder(speech_folder, "speech", "Speech")
process_folder(noise_folder, "noise", "Noise")

print("\n\n Ολοκληρώθηκε η εξαγωγή MFCC και η αποθήκευση JSON αρχείων.\n")



# --- 2. Δημιουργία X και y από τα JSON ---
X = []
y = []

# --- Συνάρτηση που φορτώνει τα MFCCs από .json αρχεία και προσθέτει τις ετικέτες ---
def load_mfcc_jsons(json_path, label):
    for file in os.listdir(json_path):
        if file.endswith(".json"):
            full_path = os.path.join(json_path, file)
            with open(full_path, "r") as f:
                data = json.load(f)
                X.extend(data)
                y.extend([label] * len(data)) # Κάθε frame παίρνει την ίδια ετικέτα (1 για speech, 0 για noise)

# Φόρτωση Speech (1) και Noise (0)
load_mfcc_jsons(f"{output_folder}/speech", 1)
load_mfcc_jsons(f"{output_folder}/noise", 0)

X = np.array(X)
y = np.array(y)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Ετικέτες: {Counter(y)}")

#Παράδειγμα εκτέλεσης κώδικα:
# X shape: (931936, 13)
# y shape: (931936,)
# Counter({np.int64(1): 907381, np.int64(0): 24555})