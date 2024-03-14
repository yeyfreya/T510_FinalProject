import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tempfile import NamedTemporaryFile
from st_audiorec import st_audiorec

# v2
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

def main():
    st.title('Real-Time Speech Emotion Recognition')

    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []

    emotion_colors = {
        'happy': 'yellow', 'sad': '#0000FF', 'angry': 'red', 'neutral': 'grey',
        'calm': 'green', 'disgust': 'purple', 'fear': 'red', 'surprise': 'orange'
    }

    audio_data = st_audiorec()

    if st.session_state.emotion_history:
        st.write("Predicted Emotions History:")
        for emotion in st.session_state.emotion_history:
            color = emotion_colors.get(emotion, "black")
            st.markdown(f'<span style="color: {color};">{emotion}</span>', unsafe_allow_html=True)

    if audio_data is not None:
        with NamedTemporaryFile(delete=True) as tmp_file:
            tmp_file.write(audio_data)
            tmp_file.seek(0)
            audio_file = tmp_file.name
            st.audio(audio_file, format='audio/wav')

            if st.button('Analyze Emotion'):
                features = preprocess_audio(audio_file)
                prediction = model.predict(features)
                predicted_emotion = np.argmax(prediction)
                emotions = ['happy', 'sad', 'angry', 'neutral', 'calm', 'disgust', 'fear', 'surprise']
                predicted_emotion_label = emotions[predicted_emotion]
                st.session_state.emotion_history.append(predicted_emotion_label)

                # Immediately reflect the update without needing to press the button again
                st.experimental_rerun()

if __name__ == '__main__':
    main()
