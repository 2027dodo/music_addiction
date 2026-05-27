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

    # Tempo
    T, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Repetition
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    R = np.mean(librosa.autocorrelate(onset_env))

    # Energy
    rms = librosa.feature.rms(y=y)[0]
    E = np.mean(rms)

    # Danceability
    loader = es.MonoLoader(filename=file_path)
    audio = loader()
    get_danceability = es.Danceability()
    D = get_danceability(audio)

    # Flatten danceability if needed
    if hasattr(D, "__iter__"):
        D = np.mean(list(flatten(D)))

    return T, E, R, D

def flatten(seq):
    for item in seq:
        if hasattr(item, '__iter__') and not isinstance(item, (str, bytes)):
            yield from flatten(item)
        else:
            yield item

def get_audio(folder_path):
    audio_files = []
    for pattern in ["*.mp3"]:
        audio_files.extend(glob.glob(os.path.join(folder_path, pattern)))
    return audio_files

def get_available_genres(music_folder):
    genres = []
    if os.path.exists(music_folder):
        for item in os.listdir(music_folder):
            if os.path.isdir(os.path.join(music_folder, item)):
                genres.append(item)
    return sorted(genres)

if __name__ == "__main__":
    music_folder = "music_mp3"

    available_genres = get_available_genres(music_folder)
    
    # Choose genre
    print("Available genres:")
    for genre in available_genres:
        print(" ", genre)
    print()
    
    genre_input = input("Enter genre name: ").strip()
    genre_folder = os.path.join(music_folder, genre_input)  
    audio_files = get_audio(genre_folder)

    # Linear regression parameters
    beta_1 = 1.0      # Tempo scale
    beta_2 = 100.0    # Energy scale
    beta_3 = 0.01     # Repetition scale
    beta_4 = 10.0     # Danceability scale
    epsilon = 0

    results = []
    for audio_file in audio_files:
        T, E, R, D = extract_audio_features(audio_file)
        S = compute_linear_regression_model(beta_1, beta_2, beta_3, beta_4, T, E, R, D, epsilon)
        results.append((audio_file, S))

    # Print results in file order
    for i, (file_path, score) in enumerate(results, 1):
        filename = os.path.basename(file_path)
        print(f"{i:2d}. {filename}: {np.round(score, 3)}")
