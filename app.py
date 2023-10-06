import streamlit as st
from st_audiorec import st_audiorec
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
import pandas as pd
import torch
# import torchaudio
import wave
import io
from scipy.io import wavfile
import pydub
import time
import os
import atexit
import librosa

# MODEL LOADING and INITIALISATION
model = torch.jit.load("snorenetv1_small.ptl")
model.eval()
endReached = False
snore = 0
other = 0
s=0
n=16000

# Audio parameters

st.sidebar.markdown(
    """
    <div align="justify">
        <h4>ABOUT</h4>
        <p>Transform your sleep experience with the cutting-edge Snore Detector by Hypermind Labs!
        Discover the power to monitor and understand your nighttime sounds like never before.
        Take control of your sleep quality and uncover the secrets of your peaceful slumber with our innovative app.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.title('Real-Time Snore Detection App 😴')

uploaded_file = st.file_uploader("Upload Sample", type=["wav"])
if uploaded_file is not None:
    st.write("Analsysing...")
    time.sleep(2.5)
    audio, sample_rate = librosa.load(uploaded_file, sr=None)
    waveform = audio
    # Set the chunk size
    chunk_size = 16000

    # Calculate the number of chunks
    num_chunks = len(waveform) // chunk_size

    # Reshape the waveform into chunks
    waveform_chunks = np.array_split(waveform[:num_chunks * chunk_size], num_chunks)
    
    for chunk in waveform_chunks:
        input_tensor = torch.tensor(chunk).unsqueeze(0).to(torch.float32)
        result = model(input_tensor)
        if np.abs(result[0][0]) > np.abs(result[0][1]):
            other += 1
        else:
            snore += 1

    total = snore + other
    snore_percentage = (snore / total) * 100
    other_percentage = (other / total) * 100

    categories = ["Snore", "Other"]
    percentages = [snore_percentage, other_percentage]

    plt.figure(figsize=(8, 4))
    plt.barh(categories, percentages, color=['#ff0033', '#00ffee'])
    plt.xlabel('Percentage')
    plt.title('Percentage of Snoring')
    plt.xlim(0, 100)

    for i, percentage in enumerate(percentages):
        plt.text(percentage, i, f' {percentage:.2f}%', va='center')
    st.write("DONE")
    st.pyplot(plt)
    

    # # PERCENTAGE OF SNORING PLOT

    
    






