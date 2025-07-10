 AI Music Generation using LSTM
This project showcases how to generate new musical sequences using Artificial Intelligence. By training an LSTM-based deep learning model on a dataset of MIDI files, we generate original melodies and convert them back into playable MIDI audio.

- Features
MIDI Data Collection
Automatically downloads public-domain MIDI files from the web (no uploads required).

Preprocessing with music21
Extracts notes and chords from MIDI files and encodes them into numerical sequences suitable for neural networks.

Deep Learning with LSTM
Builds and trains a two-layer LSTM model using TensorFlow/Keras to learn musical patterns.

Music Generation
Generates new note sequences using the trained model, with a temperature parameter for controlling creativity.
