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

    # Initialize the list in session state if it doesn't exist
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []

    # Mapping of emotions to colors
    emotion_colors = {
        'happy': 'yellow',
        'sad': '#0000FF',  # Using hex color codes
        'angry': 'red',
        'neutral': 'grey',
        'calm': 'green',
        'disgust': 'purple',
        'fear': 'black',
        'surprise': 'orange'
    }

    # Custom Streamlit audio recorder for capturing user input
    audio_data = st_audiorec()

    # Always display the history of predicted emotions
    if st.session_state.emotion_history:
        st.write("Predicted Emotions History:")
        for emotion in st.session_state.emotion_history:
            # Use the emotion_colors dictionary to get the corresponding color
            color = emotion_colors.get(emotion, "black")  # Default color is black if emotion not found
            # Display each emotion in its corresponding color
            st.markdown(f'<span style="color: {color};">{emotion}</span>', unsafe_allow_html=True)
    else:
        st.write("No emotions predicted yet.")

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
                predicted_emotion_label = emotions[predicted_emotion]
                
                # Append the new predicted emotion to the history
                st.session_state.emotion_history.append(predicted_emotion_label)

if __name__ == '__main__':
    main()



