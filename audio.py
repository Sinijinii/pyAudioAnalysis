import subprocess
import os
import numpy as np
import librosa
from pydub import AudioSegment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

path = "C:/Users/oops5/OneDrive/Desktop/SSAFY/A409/audio/pyAudioAnalysis/pyAudioAnalysis/"

# ffmpeg 경로 설정
AudioSegment.converter = "D:/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"
AudioSegment.ffprobe = "D:/ffmpeg-master-latest-win64-gpl/bin/ffprobe.exe"

def convert_to_wav(input_path, output_path):
    command = f"D:/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe -y -i {input_path} {output_path}"
    subprocess.call(command, shell=True)

def clean_audio(y):
    y = np.nan_to_num(y)  # NaN 값을 0으로 변환
    y[np.isinf(y)] = 0  # 무한대 값을 0으로 변환
    return y

def extract_features(y, sr, n_mfcc=13):
    y = clean_audio(y)  # NaN과 무한대 값 제거
    if len(y) < 2048:  # 최소 신호 길이를 보장
        y = np.pad(y, (0, 2048 - len(y)), 'constant')

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    features = np.hstack([
        np.mean(mfccs.T, axis=0),
        np.std(mfccs.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.std(chroma.T, axis=0),
        np.mean(mel.T, axis=0),
        np.std(mel.T, axis=0),
        np.mean(contrast.T, axis=0),
        np.std(contrast.T, axis=0),
        np.mean(tonnetz.T, axis=0),
        np.std(tonnetz.T, axis=0),
        np.mean(zcr.T, axis=0),
        np.std(zcr.T, axis=0),
        np.mean(rms.T, axis=0),
        np.std(rms.T, axis=0)
    ])
    
    return features.flatten()

def segment_features(features, segment_size=100):
    num_segments = features.shape[0] // segment_size
    segments = np.array([features[i * segment_size: (i + 1) * segment_size] for i in range(num_segments)])
    return segments

def cluster_segments(segments, num_clusters=2):
    kmeans = KMeans(n_clusters=num_clusters)
    segments_flattened = segments.reshape(segments.shape[0], -1)
    labels = kmeans.fit_predict(segments_flattened)
    return labels, kmeans.cluster_centers_

def calculate_speaker_times(labels, segment_size, sr):
    unique_labels = np.unique(labels)
    speaker_times = {label: 0 for label in unique_labels}
    for label in labels:
        speaker_times[label] += segment_size / sr
    return speaker_times

def filter_non_speech_frames(y, sr, rms_threshold=0.04):
    # RMS 값 계산
    rms = librosa.feature.rms(y=y)[0]
    
    # 각 RMS 프레임의 길이 계산
    frame_length = int(len(y) / len(rms))
    
    # RMS 값 배열을 원래 오디오 배열 길이에 맞게 확장
    expanded_indices = np.repeat(rms < rms_threshold, frame_length)
    
    # 확장된 인덱스가 원래 오디오 배열보다 길면 자르기
    if len(expanded_indices) < len(y):
        expanded_indices = np.append(expanded_indices, np.zeros(len(y) - len(expanded_indices), dtype=bool))
    elif len(expanded_indices) > len(y):
        expanded_indices = expanded_indices[:len(y)]
    
    filtered_y = y[~expanded_indices]
    return filtered_y

def identify_speakers(audio_path, my_voice_features, threshold=20.0):
    y, sr = librosa.load(audio_path)
    y = clean_audio(y)  # NaN과 무한대 값 제거
    y = filter_non_speech_frames(y, sr)  # 비음성 프레임 제거

    intervals = librosa.effects.split(y, top_db=20)
    speaker_features = []
    speaker_times = []

    for interval in intervals:
        start, end = interval
        segment = y[start:end]
        features = extract_features(segment, sr)
        speaker_features.append(features)
        speaker_times.append((start, end))

    speaker_features = np.array(speaker_features)
    scaler = StandardScaler()
    speaker_features = scaler.fit_transform(speaker_features)

    labels, centers = cluster_segments(speaker_features, num_clusters=2)

    my_voice_time = 0
    other_speaker_time = 0
    speaker_durations = {i: 0 for i in range(2)}

    for sf, (start, end), label in zip(speaker_features, speaker_times, labels):
        duration = (end - start) / sr
        sf_flat = sf.flatten()
        my_voice_flat = my_voice_features.flatten()

        distance, _ = fastdtw(sf_flat.reshape(-1, 1), my_voice_flat.reshape(-1, 1), dist=euclidean)
        if distance < threshold:
            my_voice_time += duration
        else:
            speaker_durations[label] += duration

    other_speakers_time = sum(speaker_durations.values())

    return my_voice_time, other_speakers_time, len(speaker_durations) - 1

def main():
    conversation_files = [
        r"C:/Users/oops5/OneDrive/Desktop/SSAFY/A409/audio/pyAudioAnalysis/pyAudioAnalysis/conversation1.m4a",
        r"C:/Users/oops5/OneDrive/Desktop/SSAFY/A409/audio/pyAudioAnalysis/pyAudioAnalysis/conversation2.m4a"
    ]
    my_voice_file = r"C:/Users/oops5/OneDrive/Desktop/SSAFY/A409/audio/pyAudioAnalysis/pyAudioAnalysis/my.m4a"

    my_voice_wav = my_voice_file.replace('.m4a', '.wav')
    convert_to_wav(my_voice_file, my_voice_wav)
    y, sr = librosa.load(my_voice_wav)
    y = clean_audio(y)
    my_voice_features = extract_features(y, sr)

    total_other_speakers = set()
    total_conversation_time = 0

    for conv_file in conversation_files:
        wav_file = conv_file.replace('.m4a', '.wav')
        convert_to_wav(conv_file, wav_file)

        my_voice_time, other_time, other_speakers = identify_speakers(wav_file, my_voice_features)

        total_other_speakers.add(other_speakers)
        total_conversation_time += other_time

        print(f"File: {os.path.basename(conv_file)}")
        print(f"My voice time: {my_voice_time:.2f} seconds")
        print(f"Other speakers time: {other_time:.2f} seconds")
        print(f"Number of other speakers: {other_speakers}")
        print()

    print(f"Total number of other speakers across all conversations: {len(total_other_speakers)}")
    print(f"Total conversation time (excluding my voice): {total_conversation_time:.2f} seconds")

if __name__ == "__main__":
    main()
