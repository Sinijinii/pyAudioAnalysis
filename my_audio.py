import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine

path = "C:/Users/SSAFY/Desktop/A409/Audio/pyAudioAnalysis/data"

def extract_features(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

def augment_data(y, sr):
    y_repeated = np.tile(y, 5)[:5*len(y)]  # 음성을 5배로 늘림
    return [y, y_repeated]

def train_gmm_model(file_paths):
    features = []
    for file_path in file_paths:
        y, sr = librosa.load(file_path, sr=None)
        augmented_y = augment_data(y, sr)
        for y_aug in augmented_y:
            feature = extract_features(y_aug, sr)
            features.append(feature)
    
    features = np.array(features)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    n_components = min(len(features), 4)  # 데이터 포인트 수 이하로 설정
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag', n_init=3)
    gmm.fit(features)
    
    return gmm, scaler

def recognize_speaker(gmm, scaler, y, sr):
    features = extract_features(y, sr)
    features = scaler.transform([features])
    score = gmm.score(features)
    return score, features

def cosine_similarity(v1, v2):
    return 1 - cosine(v1, v2)

def analyze_conversation(conversation_file, gmm, scaler, my_voice_features, segment_duration=1.0, threshold=0.99):
    y, sr = librosa.load(conversation_file, sr=None)
    segment_length = int(segment_duration * sr)
    
    total_duration = 0
    my_voice_duration = 0
    
    for start in range(0, len(y), segment_length):
        end = min(start + segment_length, len(y))
        segment = y[start:end]
        augmented_segments = augment_data(segment, sr)
        avg_similarity = 0
        
        for aug_segment in augmented_segments:
            score, features = recognize_speaker(gmm, scaler, aug_segment, sr)
            similarities = [cosine_similarity(features.flatten(), my_voice_feature) for my_voice_feature in my_voice_features]
            avg_similarity += np.mean(similarities)
        
        avg_similarity /= len(augmented_segments)
        duration = (end - start) / sr
        
        print(f"Segment {start//segment_length + 1}: Average Cosine Similarity = {avg_similarity}")
        
        if avg_similarity >= threshold:
            my_voice_duration += duration
        total_duration += duration
    
    return my_voice_duration, total_duration

# 내 목소리로 GMM 모델 훈련
train_files = [f"{path}/my_voice1.wav", f"{path}/my_voice2.wav", f"{path}/my_voice3.wav", f"{path}/my_voice4.wav"]
gmm_model, scaler = train_gmm_model(train_files)

# 내 목소리 특징 추출
my_voice_features = []
for file_path in train_files:
    y, sr = librosa.load(file_path, sr=None)
    feature = extract_features(y, sr)
    my_voice_features.append(feature)

# 대화 파일 분석
conversation_file = f"{path}/conversation6.wav"
my_voice_duration, total_duration = analyze_conversation(conversation_file, gmm_model, scaler, my_voice_features)

print(f"내 목소리 시간: {my_voice_duration:.2f} 초")
print(f"총 대화 시간: {total_duration:.2f} 초")
print(f"내 목소리 비율: {my_voice_duration / total_duration:.2%}")
