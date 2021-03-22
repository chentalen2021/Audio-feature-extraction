#%%
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

#%%
file = "/Users/talen/Documents/Datasets/get-off-me.wav"
audio, sr = librosa.load(file)
print(audio.size)       #The number of samples in the digitalised audio

#%% Extract Zero-Crossing-Rate feature
FRAME_LENGTH = 1024
HOP_LENGTH = 512

zcr_df = librosa.feature.zero_crossing_rate(audio,frame_length=FRAME_LENGTH,hop_length=HOP_LENGTH)
zcr = zcr_df[0]
#zcr is the normalised value; the actual value of zcr is zcr*FRAME_LENGTH

#%% Plot the ZCR for the whole utterance
frames = range(len(zcr))
t = librosa.frames_to_time(frames,hop_length=HOP_LENGTH)

plt.plot(t, zcr, color='r')
plt.ylim((0,1))
plt.title("Zero Crossing Rate")
plt.show()

