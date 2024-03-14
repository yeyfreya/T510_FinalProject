import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tempfile import NamedTemporaryFile
from st_audiorec import st_audiorec
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas as pd 
import matplotlib.dates as mdates
import os
from datetime import datetime 


# Apply a seaborn style to make plots look nicer
sns.set(style="whitegrid")

# Setting up a consistent theme for matplotlib plots
plt.rcParams.update({
    'axes.titlesize': 14,  # Title size
    'axes.labelsize': 12,  # Axis label size
    'xtick.labelsize': 10,  # X-axis tick label size
    'ytick.labelsize': 10,  # Y-axis tick label size
    'axes.titlecolor': 'darkblue',  # Title color
    'axes.labelcolor': 'darkblue',  # Axis label color
})

# MODEL_PATH updated to point to your TensorFlow Lite model
MODEL_PATH = 'soundclassifier_with_metadata.tflite'

@st.cache_resource  # Updated cache decorator

def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

model = load_model()

def preprocess_audio(audio_file):
    # Load the audio file
    audio, sr = librosa.load(audio_file, sr=44100)
    # Assuming the model expects a specific input length, pad/trim the audio file to match
    audio = librosa.util.fix_length(audio, size=44032)
    # Reshape to match the model's input shape, adjust as necessary based on your model's requirements
    features = np.expand_dims(audio, axis=0).astype(np.float32)
    return features

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

def predict_emotion(features, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], features)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    return prediction

def get_contrasting_text_color(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    return 'white' if luminance < 0.5 else 'black'

def plot_emotion_distribution(emotion_history_with_timestamps, emotion_colors):
    emotion_history = [emotion for emotion, _ in emotion_history_with_timestamps]
    emotion_counts = {emotion: emotion_history.count(emotion) for emotion in set(emotion_history)}
    labels = list(emotion_counts.keys())
    sizes = list(emotion_counts.values())
    colors = [emotion_colors[emotion] for emotion in labels]
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Distribution of Predicted Emotions')
    st.pyplot(plt)


def plot_emotion_timeline(emotion_history_with_timestamps, emotion_colors):
    emotions, timestamps = zip(*emotion_history_with_timestamps)
    timestamps = pd.to_datetime(timestamps)
    unique_emotions = sorted(set(emotions))
    emotion_to_num = {emotion: i for i, emotion in enumerate(unique_emotions)}
    emotion_nums = [emotion_to_num[emotion] for emotion in emotions]
    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, emotion_nums, marker='o', linestyle='-', color='blue')
    plt.xticks(rotation=45)
    plt.yticks(range(len(unique_emotions)), unique_emotions)
    plt.title('Timeline of Predicted Emotions')
    plt.xlabel('Time')
    plt.ylabel('Emotion')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.tight_layout()
    st.pyplot(plt)

def plot_emotion_histogram(emotion_history):
    plt.figure(figsize=(10, 6))
    plt.hist(emotion_history, bins=len(set(emotion_history)), color='skyblue')
    plt.title('Histogram of Predicted Emotions')
    plt.xlabel('Emotion')
    plt.ylabel('Frequency')
    st.pyplot(plt)


def save_audio_data(audio_data, directory="saved_audio"):
    """
    Saves the audio data to a file in the specified directory.
    
    Parameters:
    - audio_data: The binary audio data to save.
    - directory: The directory where audio files will be saved. Defaults to 'saved_audio'.
    """
    # Ensure the target directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Generate a unique filename based on the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Correct usage of datetime.now()
    filename = f"{timestamp}.wav"
    filepath = os.path.join(directory, filename)
    
    # Write the audio data to a file
    with open(filepath, 'wb') as file:
        file.write(audio_data)
    
    return filepath
    

def main():
    st.title('VoiceVibe 🎙️')
    # write app description
    st.write("VoiceVibe is a real-time speech emotion recognition app that uses a pre-trained model to analyze the emotions in your voice. ")
    st.write("Simply click the button below to record your voice and see the predicted emotions. ")
    st.write("The app will also display the distribution of emotions, a timeline of predicted emotions, and a histogram of predicted emotions. ")

    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
        st.session_state.quantified_emotions = []

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
            # Save the recorded audio data
            filepath = save_audio_data(audio_data)
            st.write(f"Audio saved to {filepath}")

            with NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file.flush()
                features = preprocess_audio(tmp_file.name)
                prediction = predict_emotion(features, model)
                emotions = ['happy', 'sad', 'angry', 'neutral', 'calm', 'disgust', 'fear', 'surprise']
                predicted_emotion = emotions[np.argmax(prediction)]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.emotion_history.append((predicted_emotion, timestamp))
                st.session_state.quantified_emotions.append(emotion_values[predicted_emotion])
                st.experimental_rerun()

    if st.session_state.emotion_history:

        st.markdown("## Predicted Emotions History")

        # Calculate the number of columns. You might adjust this based on your UI preference.
        num_cols = 4  # Example: aiming for 4 items per row
        rows = [st.session_state.emotion_history[i:i + num_cols] for i in range(0, len(st.session_state.emotion_history), num_cols)]

        for idx, row in enumerate(rows):
            cols = st.columns(len(row))  # Create columns for the row
            for col, (emotion, time) in zip(cols, row):
                color = emotion_colors.get(emotion, "black")
                text_color = get_contrasting_text_color(color)
                # Use the column to display the emotion box. Adjust styling as needed.
                col.markdown(f"""
                <div style="background-color: {color}; color: {text_color}; 
                            padding: 10px; border-radius: 10px; 
                            display: flex; flex-direction: column; justify-content: center; align-items: center;
                            text-align: center;">
                    <strong>{emotion.capitalize()}</strong>
                    <div style="margin-top: 10px; font-size: 0.8rem;">{time}</div>
                </div>
                """, unsafe_allow_html=True)

            if idx < len(rows) - 1:  # Check if it's not the last row
                st.markdown("""<br>""", unsafe_allow_html=True)  # Add extra space after each row except the last one


        if st.session_state.emotion_history:

            #add blank line
            st.markdown("""<br>""", unsafe_allow_html=True)
                
            st.markdown("### 📈 Emotion Distribution")
            plot_emotion_distribution(st.session_state.emotion_history, emotion_colors)

            st.markdown("### ⏳ Emotion Timeline")
            plot_emotion_timeline(st.session_state.emotion_history, emotion_colors)

            st.markdown("### 📊 Emotion Histogram")
            plot_emotion_histogram([emotion for emotion, _ in st.session_state.emotion_history])

            st.markdown("### 📉 Mood Swing")
            # Generate and display the area chart for quantified emotions
            df = pd.DataFrame({
                'Emotion Value': st.session_state.quantified_emotions,
                'Time Step': range(len(st.session_state.quantified_emotions))
            })
            st.area_chart(df.set_index('Time Step'))

if __name__ == '__main__':
    main()
