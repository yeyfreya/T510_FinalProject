import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tempfile import NamedTemporaryFile
import datetime
import pandas as pd
import altair as alt

# Apply seaborn style
sns.set(style="whitegrid")

MODEL_PATH = 'tone_recognition.h5'

@st.cache_resource
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
        st.session_state.quantified_emotions = []  # Initialize a list to store quantified emotions

    emotion_colors = {
        'happy': '#FFD700',
        'sad': '#1E90FF',
        'angry': '#FF4500',
        'neutral': '#808080',
        'calm': '#32CD32',
        'disgust': '#9932CC',
        'fear': '#FF0000',
        'surprise': '#FFA500'
    }

    emotion_values = {
        'surprise': 4,
        'happy': 3,
        'calm': 1,
        'neutral': 0,
        'disgust': -1,
        'sad': -2,
        'fear': -3,
        'angry': -4
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
                st.session_state.quantified_emotions.append(emotion_values[predicted_emotion_label])
                st.experimental_rerun()

    if st.session_state.emotion_history:
        st.write("Predicted Emotions History:")

        # Prepare data for Altair chart
        data = pd.DataFrame({
            'Time': [datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S") for _, time in st.session_state.emotion_history],
            'Emotion': [emotion for emotion, _ in st.session_state.emotion_history]
        })

        # Create an interactive chart
        chart = alt.Chart(data).mark_circle(size=60).encode(
            x='Time:T',
            y=alt.Y('Emotion:N', sort=list(emotion_colors.keys())),
            color=alt.Color('Emotion:N', legend=None, scale=alt.Scale(domain=list(emotion_colors.keys()), range=list(emotion_colors.values()))),
            tooltip=['Time:T', 'Emotion:N']
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
        
       # Visualization function calls
        plot_emotion_distribution([emotion for emotion, _ in st.session_state.emotion_history], emotion_colors)
        plot_emotion_timeline([emotion for emotion, _ in st.session_state.emotion_history], emotion_colors)
        plot_emotion_histogram([emotion for emotion, _ in st.session_state.emotion_history])

        # Area chart for emotion over time
        df = pd.DataFrame({
            'Emotion Value': st.session_state.quantified_emotions,
            'Time Step': range(len(st.session_state.quantified_emotions))
        })
        st.area_chart(df.set_index('Time Step'))

if __name__ == '__main__':
    main()
