# cd ~/Desktop/music_addiction
# python3 music_addictiveness.py
# 우리나라 자랑거리 = 4090.956
# 탕후루 = 22719.762
# 아리랑 = 4258.208
# 아파트 = 9620.182
# adhd music(ADDADHD Intense Relief)(3hr) = 132004.66
# adhd music (0524 (1))(5min) = 3995.51
# 5 min moonlight sonata = 1657.131
# Billie Eilish Bad Guy = 12972.396
# 아이유 네모의 꿈 = 8841.882

import librosa
import numpy as np
import math

def compute_addictiveness_score(R, B, C, v, B_0, C_0, alpha_1, alpha_2, alpha_3, beta_1, beta_2, lambda_):
    tempo_component = alpha_1 * (R / (1 + np.exp(-beta_1 * (B - B_0))))
    complexity_component = alpha_2 * (1 / (1 + math.exp(-beta_2 * (C - C_0))))
    pitch_component = alpha_3 * math.exp(-lambda_ * v)
    return tempo_component + complexity_component + pitch_component

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path)

    # Tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Repetition estimate
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    R = np.mean(librosa.autocorrelate(onset_env))

    # Chord complexity
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chord_complexity = np.mean(np.std(chroma, axis=1))

    # Pitch variability
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[magnitudes > np.median(magnitudes)]
    pitch_variability = np.std(pitch_vals) if len(pitch_vals) > 0 else 0

    return R, tempo, chord_complexity, pitch_variability

#Run
if __name__ == "__main__":
    audio_file = "Square's dream (네모의 꿈).mp3"

    R, B, C, v = extract_audio_features(audio_file)

    # Parameters
    B_0 = 100
    C_0 = 5
    alpha_1 = 1.0
    alpha_2 = 1.5
    alpha_3 = 0.5
    beta_1 = 0.1
    beta_2 = 0.2
    lambda_ = 0.8

    A = compute_addictiveness_score(R, B, C, v, B_0, C_0, alpha_1, alpha_2, alpha_3, beta_1, beta_2, lambda_)
    print("🎵 Addictiveness Score from Audio:", np.round(A, 3))
