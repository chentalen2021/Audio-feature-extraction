#%%
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

#%%
file = "/Users/talen/Documents/Datasets/get-off-me.wav"
audio, sr = librosa.load(file)
print(audio.shape)       #The number of samples in the digitalised audio

#%% Calculate the spectral centroids
FRAME_SIZE = 1024
HOP_SIZE = 512

sc = librosa.feature.spectral_centroid(y=audio,sr=sr,n_fft=FRAME_SIZE,hop_length=HOP_SIZE)[0]
print(sc)

#%% Visualise the spectral centroids
frames = range(len(sc))
t = librosa.frames_to_time(frames=frames,sr=sr,hop_length=HOP_SIZE,n_fft=FRAME_SIZE)

plt.plot(t, sc, color='r')
plt.xlabel("Time")
plt.ylabel("Spectral centroid")
plt.title("Spectral Centroids")
plt.show()

#%% Calculate the bandwidth
bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr,n_fft=FRAME_SIZE,hop_length=HOP_SIZE)[0]
print(bw.shape)

#%% Visualise the bandwidth
plt.plot(t, bw, color='b')
plt.xlabel("Time")
plt.ylabel("spectral bandwidth")
plt.title("Spectral Bandwidth")
plt.show()












