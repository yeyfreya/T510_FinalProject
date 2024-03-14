# VoiceVibe🎙️  
## <span style="color:Green;"> A Real-time Speech Emotion Recognition App with Customized ML Model</span>
 
#### Team member: Freya Yu, Gwen Zhang
 
## Technologies used

### 1. TensorFlow and Keras for Model Building and Prediction 

- **TensorFlow Lite Interpreter**: The TensorFlow Lite Interpreter is used to load our pre-trained TensorFlow Lite model for emotion recognition from audio data.

- **Keras**: An open-source neural network library written in Python, designed to enable fast experimentation with deep neural networks. It runs on top of TensorFlow and is integrated directly into TensorFlow as tf.keras. It simplifies many aspects of creating and compiling a model, adding layers, and setting up multi-input/output models.

###  2. Librosa for Audio Processing

- **Librosa**: We utilized `librosa` to load audio files, fix the audio length to the size that the model expects, and to extract audio features.


### 3. Machine Learning Model Workflow 

The core workflow in our app involves:

- Capturing audio input from the user via the Streamlit interface.
Preprocessing the audio input using Librosa to extract relevant features.
- Using the pre-trained TensorFlow/Keras model to predict the emotion from the extracted audio features.
- Visualizing the predictions and providing insights into the detected emotions over time.

### 4. Others

- **Streamlit**: It serves as the backbone of our web app, providing a framework to create our user interface, handle interactions with the user (such as button clicks and audio recording), and display data and plots.
 
- **NumPy**: We used NumPy for numerical operations on arrays, such as reshaping arrays to match the input requirements of the machine learning model.
 
- **st_audiorec**: This is a Streamlit component that allows audio recording and downloading within the Streamlit app.
 
- **Matplotlib**: We used `Matplotlib` for creating visualizations such as pie charts, line plots, and histograms.
 
- **Seaborn**: We used `Seaborn` to enhance the visualizations, making them more aesthetically pleasing with more color palettes.
 
- **Pandas**: We used Pandas for organizing the prediction results into a structured format, which can then be easily plotted and for handling timestamps and numerical data together.
 
 
## What problems you are trying to solve

The problem we want to solve is enabling automatic mood tracking and prediction using speech tone analysis for bipolar disorder individuals. This aims to understand emotional states by listening to the nuances in someone's voice, acting as a support mechanism.
 
## How to run
 
Open the terminal and run the following commands:

```
pip install -r requirements.txt
streamlit run app.py
```
 
## Reflections
#### 1. What you learned

We discovered that Streamlit can be used to create web applications that visualize data in real-time. This involves setting up our app to fetch and display data that updates dynamically, offering us insights into changing datasets as they evolve.

We learned how to integrate machine learning models into Streamlit applications, enabling interactive and practical demonstrations of the models' capabilities. This includes loading pre-trained models and using them to make predictions based on user input.

Beyond static predictions, we explored the concept of visualizing machine learning model predictions in real-time within a Streamlit app. This approach enhances user engagement by providing instant feedback based on the model's analysis of live data inputs.

We inquired about implementing real-time voice input capabilities within Streamlit apps. While Streamlit does not natively support real-time audio capture and processing, we learned about potential workarounds involving custom JavaScript for audio capture and processing the audio data on the server side.

A specific solution for capturing audio within Streamlit apps was introduced through the streamlit_audio_recorder component by Stefan Rummer. This component enables audio recording directly in the app, facilitating use cases like emotion detection from speech.

We learned the process for exporting machine learning models trained in Google Colab to be used in external applications, such as Streamlit apps. This includes saving the model, downloading it from Colab, and loading it into the Streamlit app for inference.
 
#### 2. What questions/problems did you face?
1. When we deploy our ML model in streamlit, we find the accuracy is very off while our ML model itself has a decent accuracy. The model we trained in colab notebook has a 0.75 accuracy but when transitioning to streamlit, the prediction is very inaccurate. The model we exported from Teachable Machine is slightly better when transitioning into streamlit web app.
   
2. When training the model, we tried to use CNN + LSTM as our model and used large datasets and we ran into insufficient RAM issue in Colab Notebook.
