import subprocess
import os
import numpy as np
import librosa
from pydub import AudioSegment
from uisrnn import UISRNN
import argparse

path = "C:/Users/oops5/OneDrive/Desktop/SSAFY/A409/py/pyAudioAnalysis/data/"

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
    ]).astype(np.float32)  # float32 타입으로 변환

    return features.flatten()

def prepare_training_data(my_voice_files):
    features_list = []
    cluster_ids = []
    for i, my_voice_file in enumerate(my_voice_files):
        my_voice_wav = my_voice_file.replace('.m4a', '.wav')
        convert_to_wav(my_voice_file, my_voice_wav)
        y, sr = librosa.load(my_voice_wav)
        y = clean_audio(y)
        features = extract_features(y, sr)
        features_list.append(features)
        cluster_ids.append(i)  # 정수로 변환

    return np.array(features_list, dtype=np.float32), np.array(cluster_ids, dtype=np.int32)

def pack_sequence(sequences, labels):
    lengths = np.array([len(seq) for seq in sequences])
    max_length = max(lengths)

    # sequences를 패딩하여 동일한 길이로 맞춤
    padded_sequences = np.zeros((len(sequences), max_length), dtype=np.float32)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq

    labels = np.array(labels, dtype=np.float32)  # float32 타입으로 변환
    return padded_sequences, labels

def identify_speakers(audio_path, uisrnn_model):
    y, sr = librosa.load(audio_path)
    y = clean_audio(y)  # NaN과 무한대 값 제거

    features = extract_features(y, sr)
    features = features.reshape(1, -1)  # 2D 배열로 변환
    features = features.astype(np.float32)  # 데이터 타입 변환

    predicted_label = uisrnn_model.predict(features)
    return predicted_label

def analyze_conversation(conversation_file, uisrnn_model):
    wav_file = conversation_file.replace('.m4a', '.wav')
    convert_to_wav(conversation_file, wav_file)

    y, sr = librosa.load(wav_file)
    y = clean_audio(y)

    window_size = sr * 10  # 10초 단위로 분석
    steps = range(0, len(y), window_size)
    
    speaker_times = {}
    
    for i in steps:
        y_window = y[i:i + window_size]
        features = extract_features(y_window, sr)
        features = features.reshape(1, -1)  # 2D 배열로 변환
        features = features.astype(np.float32)  # 데이터 타입 변환

        predicted_label = uisrnn_model.predict(features)
        speaker = predicted_label[0]

        if speaker not in speaker_times:
            speaker_times[speaker] = 0
        speaker_times[speaker] += 10  # 10초 추가

    return speaker_times

def main():
    # Step 1: 목소리 학습 모델 생성
    my_voice_files = [
        f"{path}/my_voice1.m4a",
        f"{path}/my_voice2.m4a",
        f"{path}/my_voice3.m4a",
        f"{path}/my_voice4.m4a",
        f"{path}/my_voice5.m4a"
    ]

    train_features, train_cluster_ids = prepare_training_data(my_voice_files)

    # train_features와 train_cluster_ids가 올바르게 생성되었는지 확인
    print(f"Train Features Shape: {train_features.shape}")
    print(f"Train Cluster IDs Shape: {train_cluster_ids.shape}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--observation_dim', type=int, default=train_features.shape[1])
    parser.add_argument('--rnn_hidden_size', type=int, default=128)
    parser.add_argument('--rnn_depth', type=int, default=1)
    parser.add_argument('--rnn_dropout', type=float, default=0.2)
    parser.add_argument('--crp_alpha', type=float, default=1.0)
    parser.add_argument('--sigma2', type=float, default=None)
    parser.add_argument('--transition_bias', type=float, default=None)
    parser.add_argument('--verbosity', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--train_iteration', type=int, default=1000)
    parser.add_argument('--learning_rate_half_life', type=int, default=1000)
    parser.add_argument('--num_permutations', type=int, default=10)
    parser.add_argument('--regularization_weight', type=float, default=1e-5)
    parser.add_argument('--sigma_alpha', type=float, default=1.0)
    parser.add_argument('--sigma_beta', type=float, default=0.01)
    parser.add_argument('--grad_max_norm', type=float, default=5.0)
    parser.add_argument('--enforce_cluster_id_uniqueness', type=bool, default=True)
    args = parser.parse_args()

    uisrnn_model = UISRNN(args)

    # 학습 데이터 시퀀스 패킹
    packed_train_sequence, rnn_truth = pack_sequence(train_features, train_cluster_ids)
    uisrnn_model.fit_concatenated(packed_train_sequence, rnn_truth, parser.parse_args())  # 복사본 사용

    # 모델 저장
    uisrnn_model.save('my_voice_model.uisrnn')

    # Step 2: 대화 음성 파일 분석
    conversation_files = [
        f"{path}/conversation1.m4a",
        f"{path}/conversation3.m4a"
    ]

    uisrnn_model = UISRNN.load('my_voice_model.uisrnn')  # 저장된 모델 불러오기

    for conv_file in conversation_files:
        speaker_times = analyze_conversation(conv_file, uisrnn_model)
        print(f"File: {os.path.basename(conv_file)}")
        for speaker, time in speaker_times.items():
            print(f"Speaker {speaker}: {time} seconds")
        print()

if __name__ == "__main__":
    main()
