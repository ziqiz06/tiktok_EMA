import librosa
import numpy as np
import pandas as pd
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
videos_dir = os.path.join(base, "videos")

# Build a map: video_id -> file path (skip .wav files)
video_files = {
    os.path.splitext(f)[0]: os.path.join(videos_dir, f)
    for f in os.listdir(videos_dir)
    if not f.endswith(".wav")
}

# Load pilot sample and filter to URLs that were actually downloaded
pilot = pd.read_csv(os.path.join(base, "pilot_sample_events.csv"))
pilot = pilot.dropna(subset=["url"])

unique_videos = [
    url for url in pilot["url"].unique()
    if url.rstrip("/").split("/")[-1] in video_files
]
print(f"Found {len(unique_videos)} pilot URLs with downloaded video files")


def summarize_feature(x, prefix, features):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    for j in range(x.shape[0]):
        series = x[j]
        features[f"{prefix}_{j+1}_mean"] = float(np.mean(series))
        features[f"{prefix}_{j+1}_std"]  = float(np.std(series))
        if len(series) > 1:
            features[f"{prefix}_{j+1}_volatility"] = float(np.std(np.diff(series)))
        else:
            features[f"{prefix}_{j+1}_volatility"] = np.nan


feature_rows = []

for url in unique_videos:
    video_id = url.rstrip("/").split("/")[-1]
    file_path = video_files[video_id]
    print("Processing:", url)

    try:
        y, sr = librosa.load(file_path, sr=22050)

        rms       = librosa.feature.rms(y=y)
        zcr       = librosa.feature.zero_crossing_rate(y=y)
        centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast  = librosa.feature.spectral_contrast(y=y, sr=sr)

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        chroma     = librosa.feature.chroma_stft(y=y, sr=sr)
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz    = librosa.feature.tonnetz(y=y_harmonic, sr=sr)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        non_silent_intervals = librosa.effects.split(y)
        non_silent_samples   = sum(end - start for start, end in non_silent_intervals)
        non_silence_ratio    = non_silent_samples / len(y) if len(y) > 0 else np.nan

        features = {
            "url":               url,
            "tempo":             float(np.asarray(tempo).squeeze()),
            "beat_count":        int(len(beat_frames)),
            "non_silence_ratio": float(non_silence_ratio),
        }

        summarize_feature(rms,       "rms",       features)
        summarize_feature(zcr,       "zcr",       features)
        summarize_feature(centroid,  "centroid",  features)
        summarize_feature(rolloff,   "rolloff",   features)
        summarize_feature(bandwidth, "bandwidth", features)
        summarize_feature(mfcc,      "mfcc",      features)
        summarize_feature(contrast,  "contrast",  features)
        summarize_feature(chroma,    "chroma",    features)
        summarize_feature(tonnetz,   "tonnetz",   features)

        feature_rows.append(features)
        print("  Success")

    except Exception as e:
        print("  Feature extraction failed:", e)

features_df = pd.DataFrame(feature_rows)

# Merge back with pilot event info
df_final = pilot.merge(features_df, on="url", how="left")
df_final.to_csv(os.path.join(base, "section5.1", "df_final.csv"), index=False)
print(f"\nSaved {len(features_df)} videos with features → section5.1/df_final.csv")
