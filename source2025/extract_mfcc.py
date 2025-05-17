import librosa

def extract_mfcc(file_path, sr=16000, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr)  #16.000 δείγματα ανά δευτερόλεπτο.
    frame_length = int(0.025 * sr)   #frame:  25 ms = 400 δείγματα
    hop_length = int(0.010 * sr)    #Step: 10 ms = 160 δείγματα
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=frame_length, hop_length=hop_length)
    return mfcc.T

file_path = 'C:/Users/georg/Downloads/train/speech/us-gov/speech-us-gov-0005.wav'
mfcc = extract_mfcc(file_path)

print("MFCC shape:", mfcc.shape)  # π.χ. (600, 13)

#output: MFCC shape:(2678, 13)
#Το αρχείο σου περιέχει 2679 frames (δηλαδή κομματάκια των 25ms),
#και για κάθε frame εξήχθησαν 13 MFCC χαρακτηριστικά (τυπική τιμή).