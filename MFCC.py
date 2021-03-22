#%%
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

#%%
file = "/Users/talen/Documents/Datasets/get-off-me.wav"
audio, sr = librosa.load(file)
print(audio.shape)       #The number of samples in the digitalised audio

#%% Extract MFCCs
mfccs = librosa.feature.mfcc(audio, n_mfcc=13, sr=sr, window=False, n_fft=1024, hop_length=None)
mfccs.shape     # the number of MFCCs * the number of frames

#%% Visualise the MFCCs
librosa.display.specshow(mfccs, x_axis="time", sr=sr)
plt.colorbar(format="%+2f")
plt.title("MFCCs")
plt.show()

#%% Calculate the delta and the delta of delta of MFCCs
delta_mfccs = librosa.feature.delta(mfccs)
delta_double_mfccs = librosa.feature.delta(mfccs, order=2)      #order=2: calculate the second derivative
print(delta_mfccs.shape)
print(delta_double_mfccs.shape)

#%% Visualise the delta of MFCCs
librosa.display.specshow(delta_mfccs, x_axis="time", sr=sr)
plt.colorbar(format="%+2f")
plt.title("the delta of MFCCs")
plt.show()

#%% Visualise the delta of delta of MFCCs
librosa.display.specshow(delta_double_mfccs, x_axis="time", sr=sr)
plt.colorbar(format="%+2f")
plt.title("the delta of delta of MFCCs")
plt.show()

#%% Concatenate the MFCCs and the delta of MFCCs
mfccs_vector = np.concatenate((mfccs,delta_mfccs,delta_double_mfccs),axis=0)
print(mfccs_vector.shape)

