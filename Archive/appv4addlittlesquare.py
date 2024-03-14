import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tempfile import NamedTemporaryFile
from st_audiorec import st_audiorec
import matplotlib.pyplot as plt
import seaborn as sns
import datetime  # Ensure this import is included for the timestamp

# Apply seaborn style
sns.set(style="whitegrid")

MODEL_PATH = 'tone_recognition.h5'

@st.cache_resource  # Updated cache decorator
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

# Retaining the enhanced visualization functions here (plot_emotion_distribution, plot_emotion_timeline, plot_emotion_histogram)

def get_contrasting_text_color(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    return 'white' if luminance < 0.5 else 'black'

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
        'happy': '#FFD700',    # Gold
        'sad': '#1E90FF',      # DodgerBlue
        'angry': '#FF4500',    # OrangeRed
        'neutral': '#808080',  # Grey
        'calm': '#32CD32',     # LimeGreen
        'disgust': '#9932CC',  # DarkOrchid
        'fear': '#FF0000',     # Red
        'surprise': '#FFA500'  # Orange
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
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.emotion_history.append((predicted_emotion_label, timestamp))
                
                # Immediately reflect the update without needing to press the button again
                st.experimental_rerun()

    if st.session_state.emotion_history:
        st.write("Predicted Emotions History:")
        for emotion, time in st.session_state.emotion_history:
            color = emotion_colors.get(emotion, "black")
            text_color = get_contrasting_text_color(color)
            # Ensure the box HTML template creates a square-like appearance
            box_html = f"""
            <div style="background-color: {color}; color: {text_color}; 
                        padding: 20px; margin: 10px 0; border-radius: 10px; width: 100px; height: 100px; 
                        display: flex; flex-direction: column; justify-content: center; align-items: center;
                        text-align: center;">
                <strong>{emotion.capitalize()}</strong>
                <div style="margin-top: 10px; font-size: 0.8rem;">{time}</div>
            </div>
            """
            st.markdown(box_html, unsafe_allow_html=True)

        # Visualization function calls
        plot_emotion_distribution([emotion for emotion, _ in st.session_state.emotion_history], emotion_colors)
        plot_emotion_timeline([emotion for emotion, _ in st.session_state.emotion_history], emotion_colors)
        plot_emotion_histogram([emotion for emotion, _ in st.session_state.emotion_history])


if __name__ == '__main__':
    main()
