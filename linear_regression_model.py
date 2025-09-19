# cd ~/Desktop/music_addiction
# python3 music_addictiveness.py

import librosa
import numpy as np
import glob
import os
import essentia.standard as es

def compute_linear_regression_model(beta_1, beta_2, beta_3, beta_4, T, E, R, D, epsilon):
    score = (beta_1 * T) + (beta_2 * E) + (beta_3 * R) + (beta_4 * D) + epsilon
    return score

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path)

    T, _ = librosa.beat.beat_track(y=y, sr=sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    R = np.mean(librosa.autocorrelate(onset_env))

    rms = librosa.feature.rms(y=y)[0]
    E = np.mean(rms)

    loader = es.MonoLoader(filename=file_path)
    audio = loader()
    get_danceability = es.Danceability()
    D = get_danceability(audio)

    return T, E, R, D

def flatten(seq):
    for item in seq:
        if hasattr(item, '__iter__') and not isinstance(item, (str, bytes)):
            yield from flatten(item)
        else:
            yield item

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
    
    print("Available genres:")
    for genre in available_genres:
        print(" ", genre)
    print()
    
    genre_input = input("Enter genre name: ").strip()
    
    selected_genre = genre_input
    genre_folder = os.path.join(music_folder, selected_genre)  
    audio_files = get_audio(genre_folder)

    beta_1 = 1.0      # Tempo scale
    beta_2 = 100.0    # Energy scale
    beta_3 = 0.01     # Repetition scale
    beta_4 = 10.0     # Danceability scale
    epsilon = 0
    
    results = []

    for i, audio_file in enumerate(audio_files, 1):
        T, E, R, D = extract_audio_features(audio_file)

        flat_D = list(flatten(D)) if hasattr(D, "__iter__") else [D]
        D_scalar = np.mean(flat_D)

        S = compute_linear_regression_model(beta_1, beta_2, beta_3, beta_4,
                                            T, E, R, D_scalar, epsilon)
        results.append((audio_file, S))

    # Rescale final scores to 0–100
    epsilon_small = 1e-3
    scores = [score for _, score in results]
    min_score = min(scores)
    max_score = max(scores)
    scaled_results = []
    for file_path, score in results:
        scaled_score = 100 * (score - min_score + epsilon_small) / (max_score - min_score + epsilon_small)
        scaled_results.append((file_path, scaled_score))

    for i, (file_path, score) in enumerate(scaled_results, 1):
        filename = os.path.basename(file_path)
        print(f"{i:2d}. {filename}: {np.round(score, 1)}")