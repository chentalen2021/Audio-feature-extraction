#%%
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd

#%%
file = "/Users/talen/Documents/Datasets/get-off-me.wav"
audio, sr = librosa.load(file)
print(audio.size)       #The number of samples in the digitalised audio

#%% Extract Root-Mean-Square-Energy feature
FRAME_LENGTH = 1024
HOP_LENGTH = 512

rmse_df = librosa.feature.rms(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
rmse = rmse_df[0]      #The rmse is a two-dimensional array, take the first element containing the whole data

#%% Plot the RMSE for the whole utterance
frames = range(len(rmse))
t=librosa.frames_to_time(frames,hop_length=HOP_LENGTH)

librosa.display.waveplot(audio, alpha=0.5)
plt.plot(t,rmse,color='r')
plt.ylim((-1,1))
plt.title("The Root-Mean-Square-Energy")
plt.show()


#%% *Extract the RMSE manually
def rmse(signal, frame_length,hop_length):
    rmse=[]

    for i in range(0,len(signal),hop_length):
        rmse_current_frame = np.sqrt(np.sum(signal[i:i+frame_length]**2)/frame_length)
        rmse.append(rmse_current_frame)

    return np.array(rmse)

rmse_manual = rmse(audio,frame_length=FRAME_LENGTH,hop_length=HOP_LENGTH)






