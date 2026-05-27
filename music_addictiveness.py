import librosa
import numpy as np
import math
import glob
import os

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

def get_audio(folder_path):
    audio_type = ["*.mp3"]
    audio_files = []

    for type in audio_type:
        audio_files.extend(glob.glob(os.path.join(folder_path, type)))

    return audio_files
    
def get_available_genres(music_folder):
    genres = []
    if os.path.exists(music_folder):
        for item in os.listdir(music_folder):
            item_path = os.path.join(music_folder, item)
            if os.path.isdir(item_path):
                genres.append(item)
    
    return sorted(genres)


if __name__ == "__main__":
    music_folder = "music_mp3"

    available_genres = get_available_genres(music_folder)
    
    audio_files = get_audio(music_folder)

    #choose genre
    print("Available genres:")
    for genre in available_genres:
        print(" ", genre)
    print()
    
    genre_input = input("Enter genre name: ").strip()
    
    selected_genre = genre_input
    genre_folder = os.path.join(music_folder, selected_genre)  
    audio_files = get_audio(genre_folder)
    
    # Parameters
    B_0 = 100
    C_0 = 5
    alpha_1 = 1.0
    alpha_2 = 1.5
    alpha_3 = 0.5
    beta_1 = 0.1
    beta_2 = 0.2
    lambda_ = 0.8
    
    results = []
    
    for i, audio_file in enumerate(audio_files, 1):
        
        R, B, C, v = extract_audio_features(audio_file)
        
        A = compute_addictiveness_score(R, B, C, v, B_0, C_0, alpha_1, alpha_2, alpha_3, beta_1, beta_2, lambda_)
        results.append((audio_file, A))
    
    for i, (file_path, score) in enumerate(results, 1):
        filename = os.path.basename(file_path)
        print(f"{i:2d}. {filename}: {np.round(score, 3)}")