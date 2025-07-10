# CODE-ALPHA-MUSIC-GENERATION-WITH-AI
# ✅ 1. Install needed packages
import subprocess, sys
for pkg in ["music21","tensorflow","tqdm","numpy"]:
    try: __import__(pkg)
    except: subprocess.check_call([sys.executable,"-m","pip","install",pkg])

# 2. Imports
import numpy as np, warnings
from music21 import stream, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical
warnings.filterwarnings("ignore")

# 3. Generate a simple C-major scale sequence
scale = ['C4','D4','E4','F4','G4','A4','B4','C5']
notes = scale * 50  # repeat 50 times → 400 notes

# 4. Preprocess
seq_len = 16
vocab = sorted(set(notes))
note2int = {n:i for i,n in enumerate(vocab)}
X, y = [], []
for i in range(len(notes)-seq_len):
    seq_in = [note2int[n] for n in notes[i:i+seq_len]]
    seq_out = note2int[notes[i+seq_len]]
    X.append(seq_in)
    y.append(seq_out)
X = np.array(X).reshape(len(X),seq_len,1)/len(vocab)
y = to_categorical(y,num_classes=len(vocab))

# 5. Build and train model
model = Sequential([
    LSTM(64,input_shape=(seq_len,1),return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dense(64,activation='relu'),
    Dense(len(vocab),activation='softmax')
])
model.compile(loss='categorical_crossentropy',optimizer='adam')
model.fit(X,y,epochs=8,batch_size=32,verbose=1)

# 6. Generate new notes (random seed)
pattern = list(np.random.randint(0,len(vocab),seq_len))
generated = []
for _ in range(64):  # generate 64 notes
    x = np.array(pattern).reshape(1,seq_len,1)/len(vocab)
    preds = model.predict(x,verbose=0)[0]
    idx = np.random.choice(len(vocab),p=preds)
    generated.append(vocab[idx])
    pattern.append(idx)
    pattern = pattern[1:]

# 7. Convert to MIDI
out = stream.Stream()
offset=0
for n in generated:
    newn = note.Note(n)
    newn.offset = offset
    out.append(newn)
    offset+=0.5
out.write('midi',fp='generated.mid')
print("✅ generated.mid created — no downloads needed.")

