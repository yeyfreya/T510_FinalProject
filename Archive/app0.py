import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tempfile import NamedTemporaryFile
from st_audiorec import st_audiorec

# v2
MODEL_PATH = 'tone_recognition.h5' 

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

def preprocess_audio(audio_file):
    # Load the audio file
    audio, sr = librosa.load(audio_file, sr=None)
    
    # Call the feature extraction
    features = extract_features(audio, sr)
    
    # Reshape features as needed to match the input shape of your model, e.g., (1, -1) for a single sample
    features_reshaped = np.expand_dims(features, axis=0)
    
    return features_reshaped


def extract_features(audio, sr):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(audio))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result


def main():
    st.title('Real-Time Speech Emotion Recognition')

    # Custom Streamlit audio recorder for capturing user input
    audio_data = st_audiorec()

    if audio_data is not None:
        # Save temporary audio file
        with NamedTemporaryFile(delete=True) as tmp_file:
            tmp_file.write(audio_data)
            tmp_file.seek(0)
            audio_file = tmp_file.name
        
            # Display an audio player to play user's recording
            st.audio(audio_file, format='audio/wav')

            # Preprocess and predict
            if st.button('Analyze Emotion'):
                features = preprocess_audio(audio_file)
                prediction = model.predict(features)
                predicted_emotion = np.argmax(prediction)
                
                # Map your model's integer outputs back to emotion labels
                emotions = ['happy', 'sad', 'angry', 'neutral', 'calm', 'disgust', 'fear', 'surprise']
                colors = {
                    'happy': '#FFFF00',  # Yellow
                    'sad': '#0000FF',    # Blue
                    'angry': '#FF0000',  # Red
                    'neutral': '#808080',# Grey
                    'calm': '#00FFFF',   # Cyan
                    'disgust': '#008000',# Green
                    'fear': '#800080',   # Purple
                    'surprise': '#FFA500' # Orange
                }
                
                predicted_emotion_label = emotions[predicted_emotion]
                color = colors[predicted_emotion_label]

                # Use custom HTML and CSS to display the predicted emotion with its corresponding color
                st.markdown(f"<h2 style='text-align: center; color: white; background-color: {color}; padding: 10px;'>Predicted Emotion: {predicted_emotion_label}</h2>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()

