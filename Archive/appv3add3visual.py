import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tempfile import NamedTemporaryFile
from st_audiorec import st_audiorec
import matplotlib.pyplot as plt

MODEL_PATH = 'tone_recognition.h5'

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

def preprocess_audio(audio_file):
    audio, sr = librosa.load(audio_file, sr=None)
    features = extract_features(audio, sr)
    features_reshaped = np.expand_dims(features, axis=0)
    return features_reshaped

def extract_features(audio, sr):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
    stft = np.abs(librosa.stft(audio))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    result = np.hstack((result, zcr, chroma_stft, mfcc, rms, mel))
    return result

def plot_emotion_distribution(emotion_history, emotion_colors):
    emotion_counts = {emotion: emotion_history.count(emotion) for emotion in set(emotion_history)}
    labels = list(emotion_counts.keys())
    sizes = list(emotion_counts.values())
    colors = [emotion_colors[emotion] for emotion in labels]
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Distribution of Predicted Emotions')
    st.pyplot(plt)

def plot_emotion_timeline(emotion_history, emotion_colors):
    emotion_to_num = {emotion: i for i, emotion in enumerate(set(emotion_history))}
    num_to_emotion = {i: emotion for emotion, i in emotion_to_num.items()}
    emotion_nums = [emotion_to_num[emotion] for emotion in emotion_history]
    plt.figure(figsize=(10, 4))
    plt.plot(emotion_nums, marker='o', linestyle='-', color='b')
    plt.yticks(range(len(num_to_emotion)), list(num_to_emotion.values()))
    plt.title('Timeline of Predicted Emotions')
    plt.xlabel('Time (arbitrary units)')
    plt.ylabel('Emotion')
    st.pyplot(plt)

def plot_emotion_histogram(emotion_history):
    plt.figure(figsize=(10, 6))
    plt.hist(emotion_history, bins=len(set(emotion_history)), color='skyblue')
    plt.title('Histogram of Predicted Emotions')
    plt.xlabel('Emotion')
    plt.ylabel('Frequency')
    st.pyplot(plt)

def main():
    st.title('Real-Time Speech Emotion Recognition')

    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []

    emotion_colors = {
        'happy': '#FFFF00',  
        'sad': '#0000FF',    
        'angry': '#FF0000',  
        'neutral': '#808080',
        'calm': '#008000',   
        'disgust': '#800080',
        'fear': '#FF0000',   
        'surprise': '#FFA500'
    }

    audio_data = st_audiorec()

    if st.button('Analyze Emotion'):
        if audio_data is not None:
            with NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file.flush()
                features = preprocess_audio(tmp_file.name)
                prediction = model.predict(features)
                predicted_emotion = np.argmax(prediction)
                emotions = ['happy', 'sad', 'angry', 'neutral', 'calm', 'disgust', 'fear', 'surprise']
                predicted_emotion_label = emotions[predicted_emotion]
                st.session_state.emotion_history.append(predicted_emotion_label)

                # Immediately reflect the update without needing to press the button again
                st.experimental_rerun()

    if st.session_state.emotion_history:
        st.write("Predicted Emotions History:")
        for emotion in st.session_state.emotion_history:
            color = emotion_colors.get(emotion, "black")
            st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 10px; text-align: center;'>{emotion.capitalize()}</div>", unsafe_allow_html=True)

        plot_emotion_distribution(st.session_state.emotion_history, emotion_colors)
        plot_emotion_timeline(st.session_state.emotion_history, emotion_colors)
        plot_emotion_histogram(st.session_state.emotion_history)

if __name__ == '__main__':
    main()
