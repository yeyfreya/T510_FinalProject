import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tempfile import NamedTemporaryFile
from st_audiorec import st_audiorec

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

# Function to calculate a contrasting text color
def get_contrasting_text_color(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    return 'white' if luminance < 0.5 else 'black'

def main():
    st.title('Real-Time Speech Emotion Recognition')

    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []

    emotion_colors = {
        'happy': '#FFFF00',  # yellow
        'sad': '#0000FF',    # blue
        'angry': '#FF0000',  # red
        'neutral': '#808080', # grey
        'calm': '#008000',    # green
        'disgust': '#800080', # purple
        'fear': '#FF0000',    # red
        'surprise': '#FFA500' # orange
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
            text_color = get_contrasting_text_color(color)
            box_html = f"""
            <div style="background-color: {color}; color: {text_color}; 
                        padding: 10px; margin: 10px 0; border-radius: 10px; text-align: center;">
                <strong>{emotion.capitalize()}</strong>
            </div>
            """
            st.markdown(box_html, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
