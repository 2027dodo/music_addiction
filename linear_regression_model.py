# cd ~/Desktop/music_addiction
# python3 linear_regression_model.py

import librosa
import numpy as np
import math
import essentia
import essentia.standard as es

def compute_linear_regression_model(beta_0, beta_1, beta_2, beta_3, beta_4, beta_5, T, V, A, R, D, epsilon):
    score = beta_0 + (beta_1 * T) + (beta_2 * V) + (beta_3 * A) + (beta_4 * R) + (beta_5 * D) + epsilon
    return score

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path)

    # Tempo (BPM)
    T, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Repetition
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    R = np.mean(librosa.autocorrelate(onset_env))

    # Valence, arousal
    # mer = es.MusicEmotionRecognition()
   #  V, A = mer(file_path)
    V=0
    A=0

    #Danceability
    loader = es.MonoLoader(filename=file_path)
    audio = loader()
    get_danceability = es.Danceability()
    D = get_danceability(audio)

    return R, T, V, R, D

#make scalar
def flatten(seq):
    for item in seq:
        if hasattr(item, '__iter__') and not isinstance(item, (str, bytes)):
            yield from flatten(item)
        else:
            yield item


if __name__ == "__main__":
    audio_file = "Sangonomiya Kokomi Theme  Trailer Soundtrack (Looped) [Low Quality]   Genshin Impact [2.1].mp3"

    R, T, V, A, D = extract_audio_features(audio_file)

    # Convert to scalars
    T_scalar = np.mean(T)
    V_scalar = np.mean(V)
    A_scalar = np.mean(A)
    R_scalar = np.mean(R)

    flat_D = list(flatten(D))
    D_scalar = np.mean(flat_D)

    # Parameters
    beta_0 = 0
    beta_1 = 0.0014
    beta_2 = 0.0989
    beta_3 = 0.0280
    beta_4 = 0.2702
    beta_5 = 0.2035
    epsilon = 0

    S = compute_linear_regression_model(beta_0, beta_1, beta_2, beta_3, beta_4, beta_5,
                                        T_scalar, V_scalar, A_scalar, R_scalar, D_scalar, epsilon)

    print("Linear Regression Model Score:", np.round(S, 3))

