#%% Import the libraries for SER; the librosa is the main one for audio analysis
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import json
from PyEMD import EMD
import opensmile
import warnings

warnings.filterwarnings(action="ignore")

#%% Prepare the downloaded IEMOCAP dataset for SER

#Deal with the IEMOCAP metadata to gain the file paths of the improvised speeches in the four desired emotion classes
#Read the metadata about the dataset
df_descri=pd.read_csv("/Users/talen/Documents/Datasets/IEMOCAP/iemocap_metadata.csv")

#Only select the improvised samples to create a description file
df_impro=df_descri[df_descri["method"]=="impro"]

#Replace the default file-path with the local file-path after downloaded in the author's computer
new_paths = []

    #Gain the old paths from "path" column of the description file
old_paths = df_impro["path"].map(str)
for old_path in old_paths:
    #Extract the file names
    path_list = str(old_path).split("/")
    file_name = path_list[-1]

    #Concatenate the filename with the local folder path and saved in new_paths variable
    new_path = "/Users/talen/Documents/Datasets/IEMOCAP/Data/" + file_name
    new_paths.append(new_path)

#Replace the old paths with the new paths in the description file
df_impro.loc[:,["path"]]=new_paths

#Select the data about the angry, happy, sad, neutral emotions from the description file
df_ang = df_impro[df_impro["emotion"]=="ang"]
df_hap = df_impro[df_impro["emotion"]=="hap"]
df_sad = df_impro[df_impro["emotion"]=="sad"]
df_neu = df_impro[df_impro["emotion"]=="neu"]

#Concatenate the data of the four emotions
df_final = pd.concat([df_ang, df_hap, df_sad, df_neu])
df_final.shape
#%% Define the functions for preprocessing and feature extraction

#create a variable for restoring the LLDs, smfcc, their corresponding emotion classes,
#and the number of segments per signal
#emotion classes: ang -> 0, hap -> 1, sad -> 2, neu -> 3

Audio_features_final = {
    "1": {
        "LLDs":[], "Log-Mel-spectram":[], "smfcc":[], "class":[]    #n_segments_per_signal
    },
    "2": {
        "LLDs":[], "Log-Mel-spectram":[], "smfcc":[], "class":[]
    },
    "3": {
        "LLDs":[], "Log-Mel-spectram":[], "smfcc":[], "class":[]
    },
    "4": {
        "LLDs":[], "Log-Mel-spectram":[], "smfcc":[], "class":[]
    },
    "5": {
        "LLDs":[], "Log-Mel-spectram":[], "smfcc":[], "class":[]
    }
}


#Sampling and quantising the raw audio file into the digital signal
def Sampling_and_quantising(file_path):
    audiofile = file_path

    #Sampling and quantising the audio file into digital signals with the sampling rate of 16kHz
    signal, sr = librosa.load(audiofile, sr=16000)

    return signal, sr


#Framing and windowing the audio signal with specified window-width and sliding
def Framing_signal(signal,window_width, sliding, sr):
    #Framing the signal, with the frame size of 25ms and sliding of 10ms
    framesize = int((window_width/1000) * sr)
    slide = int((sliding/1000) * sr)

    frames = librosa.util.frame(signal, frame_length=framesize, hop_length=slide, axis=0)

    #Create Hamming window
    window = librosa.filters.get_window(window="ham",Nx=framesize,fftbins=True)

    #Apply the window function on each frame
    windowed_frames = []
    for frame in frames:
        windowed_frame = frame*window
        windowed_frames.append(windowed_frame)

    return windowed_frames, framesize


#Create signal segments for computing the segment-level features
def Frames_to_segments(frames, n_frames_per_segment, slide):
    segments_list = []

    n_segments = int((len(frames)-n_frames_per_segment)/slide + 1)

    for s in range(n_segments):
        segment_start = s*slide
        segment_end = segment_start+n_frames_per_segment
        seg = frames[segment_start : segment_end]

        #Concatenate the frame-level coefficients in each segment
        segment_tempo = seg[0]
        for l in range(1, len(seg)):
            segment_tempo = np.concatenate((segment_tempo, seg[l]))

        segments_list.append(segment_tempo)

    segments = np.array(segments_list)

    return segments


# Extract the LLDs of ComParE_2016 by openSMILE, except for the mfcc-related data
def Gain_LLDs(signal, sr, session, emotion):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    LLDs = smile.process_signal(signal, sr)

    drop_criteria = ["mfcc" in x for x in LLDs.columns]

    drop_indices = []
    for index in range(len(drop_criteria)):
        if drop_criteria[index]:
            drop_indices.append(LLDs.columns[index])

    LLDs.drop(labels=drop_indices, axis=1, inplace=True)

    values = LLDs.values[0]

    # Restore the LLDs in the Audio_features dictionary
    values = values.tolist()

    return values


# Compute the Log-Mel-spectrogram
def Gain_Log_Mel_spectrogram(signal, sr, n_fft, n_mels, window, win_length, hop_length,
                             save_folder, session, emotion, count):
    # Compute the Log-Mel-Spectrogram for each segment
    # 1.Compute the spectrogram for each segment by 1024 point short-time FT
    stft = librosa.core.stft(y=signal, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)
    spectrogram = np.abs(stft)

    # 2.Compute the mel-spectrogram by applying the 40 mel filter banks on the spectrogram coefficient
    mel_spectrogram = librosa.feature.melspectrogram(sr=sr, S=spectrogram, n_mels=n_mels)

    # 3.Compute the logarithm of the mel-spectrogram coefficient
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    # Transpose the spectrogram to denote its rows as the time, and columns as the log-mel-spectrogram coefficient
    log_mel_spectrogram = log_mel_spectrogram.T

    #     #4.Store the graphs of log-mel-spectrograms
    #     librosa.display.specshow(log_mel_spectrogram, x_axis="time", y_axis="log")
    #     ax = plt.gca()
    #     ax.axes.xaxis.set_visible(False)
    #     ax.axes.yaxis.set_visible(False)
    #     plt.savefig("/Users/talen/Desktop/"+save_folder+"/"+session+"/"+emotion+"."+str(count)+".jpg")
    #     print("Spectrogram of segment {0} of the audio of emotion {1} in session {2} is archived!".format(count,emotion,session))

    # 4'.Store the log-mel-spectrum coefficients in the Audio_features dictionary
    log_mel_spectrogram = log_mel_spectrogram.tolist()
    # Audio_features_final[session]["Log-Mel-spectram"].append(log_mel_spectrogram)
    return log_mel_spectrogram

# Compute the MFCCs
def Gain_MFCCs(signal, sr, n_fft, n_mels, n_mfcc, session, emotion):
    smfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, dct_type=2, norm="ortho",
                                  n_mels=n_mels, n_fft=n_fft)

    # Transpose the SMFCCs so as the row denotes the time
    smfccs = smfccs.T

    # Store the SMFCCs and their emotion labels under the respective path in the Audio_features dictionary
    smfccs = smfccs.tolist()
    # Audio_features_final[session]["smfcc"].append(smfccs)
    return smfccs

#%% Calculate segment-level features from raw audio signals
def Extract_segment_level_features(current_row):
    global h, count1, n_segments_per_signal

    # Gain the audio file path, together with its emotion classes, and session path
    audiofile = str(current_row["path"])
    emotion = str(current_row["emotion"])
    session = str(current_row["session"])

    # Sampling and quantising the raw audio into the digital signal
    signal, sr = Sampling_and_quantising(audiofile)

    # Channel 1: extracting features from original signal
    # Framing and windowing, with Hamming-window of 25ms and slide of 10ms
    frames1, _ = Framing_signal(signal=signal, window_width=25, sliding=10, sr=sr)
    # Create segments with the size of 30 frames and 10 frames overlapping (or 20 frames sliding)
    segments1 = Frames_to_segments(frames=frames1, n_frames_per_segment=30, slide=20)


    #Check the length of the audio signal
    audio_length = librosa.get_duration(signal, sr=sr)
    No_samples_in_3s = sr * 3

    signals_aligned = []
    if audio_length < 3:
        #Padding the signal if it is less than 3s
        signal_padded = librosa.util.fix_length(data=signal, size=No_samples_in_3s)
        signals_aligned.append(signal_padded)

    # Extract the segment-level LLDs with their functionals from the original signal by openSMILE
    LLDs = []
    for segment1 in segments1:
        llds = Gain_LLDs(signal=segment1, sr=sr, session=session, emotion=emotion)
        LLDs.append(llds)

    Audio_features_final[session]["LLDs"].append(LLDs)

    if emotion == "ang":
        Audio_features_final[session]["class"].append(0)
    elif emotion == "hap":
        Audio_features_final[session]["class"].append(1)
    elif emotion == "sad":
        Audio_features_final[session]["class"].append(2)
    else:
        Audio_features_final[session]["class"].append(3)

    # Channel 2: extracting features from the signal with trend removed
    # Remove signal trend by Zero-crossing detection method
    # 1.Use Empirical Mode Decomposition (EMD) method to decompose the signal into IMFs
    emd = EMD()
    IMFs = emd.emd(signal, max_imf=9)

    # 2. Select the IMFs that satisfy particular criterion
    # 2.1 Criterion analysis: ZCR_IMF_i / ZCR_IMF_1 < 0.01  =>  N_ZC_IMF_i / N_ZC_IMF_1 < 0.01, when the IMFs has the same time length
    IMFs_selected_index = []

    # 2.2 The zero crossing of the first IMF
    R_imf_1 = librosa.core.zero_crossings(IMFs[0], pad=False, zero_pos=True)
    n_R_imf_1 = sum(R_imf_1)

    for i in range(1, len(IMFs)):
        R_imf_i = librosa.core.zero_crossings(IMFs[i], pad=False, zero_pos=True)
        n_R_imf_i = sum(R_imf_i)

        # 2.3 Check the criterion
        if n_R_imf_i / n_R_imf_1 < 0.01:
            IMFs_selected_index.append(i)

        # 3. Derive the signal trend based on the selected IMFs
    T = IMFs[0]

    for index in range(1, len(IMFs_selected_index)):
        T = T + IMFs[index]

        # 4. Subtract the signal trend from the original signal
    signal_trend_removed = signal - T

    # Framing and windowing, with Hamming-window of 25ms and slide of 10ms
    frames2, framesize = Framing_signal(signal=signal_trend_removed, window_width=25, sliding=10, sr=sr)
    # print("There are {0} frames, each frame contains {1} samples".format(len(frames),framesize))

    # Create segments with the size of 30 frames and 10 frames overlapping (or 20 frames sliding)
    segments2 = Frames_to_segments(frames=frames2, n_frames_per_segment=30, slide=20)
    # Record the lenght of segments for time-step definition of RNN
    # Audio_features_final[session]["n_segments_per_signal"].append(len(segments2))

    # Extract segment-level spectrograms and SMFCCs from the signal with trend removed
    SPECTRO = []
    SMFCC = []

    for segment2 in segments2:
        # Calculate the segment-level log-mel-spectrograms by 1024 point STFT and 40 mel-filter banks
        spectro = Gain_Log_Mel_spectrogram(signal=segment2, sr=sr, n_fft=1024, n_mels=40, window=False, win_length=None,
                                 hop_length=None,
                                 save_folder="Spectrograms_segment", session=session, emotion=emotion,
                                 count=count1)
        count1 += 1
        SPECTRO.append(spectro)

        # Calculate the smfcc
        smfcc = Gain_MFCCs(signal=segment2, sr=sr, n_fft=1024, n_mels=40, n_mfcc=14, session=session, emotion=emotion)
        SMFCC.append(smfcc)

    Audio_features_final[session]["Log-Mel-spectram"].append(SPECTRO)
    Audio_features_final[session]["smfcc"].append(SMFCC)

    print("Sample {} is done!".format(h))
    h += 1


# Iterate all the speech samples
h = 1
count1 = 1

for r in range(len(df_final)):
    row = df_final.iloc[r, :]
    Extract_segment_level_features(row)

# Store the Audio_features file locally
data_path = "/Users/talen/Desktop/Audio_features_final.json"

with open(data_path, "w") as fp:
    json.dump(Audio_features_final, fp, indent=4)


#%%

# with open(data_path, "w") as fp:
#     json.dump(Audio_features, fp, indent=4)

#%% Calculate utterance-level log-mel-spectrograms from the audio signals with trend removed
# count2 = 1
#
# for r in range(len(df_final)):
#     row = df_final.iloc[r,:]
#
#     #Gain the audio file path, together with its emotion classes, and session path
#     audio = str(row["path"])
#     emotion_class = str(row["emotion"])
#     session_num = str(row["session"])
#
#     #Sampling and quantising the raw audio into the digital signal
#     signal_ori, sampling_rate = Sampling_and_quantising(audio)
#
#     # Visualise the original signal
#     librosa.display.waveplot(signal_ori, sr=sampling_rate)
#     plt.xlabel("Time")
#     plt.ylabel("Amplitude")
#     plt.title("Raw signal wave")
#
#     #Remove signal trend by Zero-crossing detection method
#         #1.Use Empirical Mode Decomposition (EMD) method to decompose the signal into IMFs
#     emd = EMD()
#     IMFs = emd.emd(signal_ori)
#
#         #2. Select the IMFs that satisfy particular criterion
#         #2.1 Criterion analysis: ZCR_IMF_i / ZCR_IMF_1 < 0.01  =>  N_ZC_IMF_i / N_ZC_IMF_1 < 0.01, when the IMFs has the same time length
#
#     IMFs_selected_index = []
#
#         #2.2 The zero crossing of the first IMF
#     R_imf_1 = librosa.core.zero_crossings(IMFs[0], pad=False, zero_pos=True)
#     n_R_imf_1 = sum(R_imf_1)
#
#     for i in range(1, len(IMFs)):
#         R_imf_i = librosa.core.zero_crossings(IMFs[i], pad=False, zero_pos=True)
#         n_R_imf_i = sum(R_imf_i)
#
#         #2.3 Check the criterion
#         if n_R_imf_i / n_R_imf_1 < 0.01:
#             IMFs_selected_index.append(i)
#
#         #3. Derive the signal trend based on the selected IMFs
#     T = IMFs[0]
#
#     for index in range(1, len(IMFs_selected_index)):
#         T = T + IMFs[index]
#
#         #4. Subtract the signal trend from the original signal
#     signal_final = signal_ori - T
#
#     librosa.display.waveplot(signal_final, sr=sampling_rate)
#     plt.xlabel("Time")
#     plt.ylabel("Amplitude")
#     plt.title("Signal wave with trend removed")
#     plt.show()
#
#     #Calculate the utterance-level log-mel-spectrograms by 512 point FFT and 40 mel-filter banks
#     #The Hamming window, with the length of 25ms and shift of 10ms, is also applied
#     #The frequency range in filter banks is set as 3000~8000
#     window_length = int((window_width / 1000) * sampling_rate)
#     window_shift = int((sliding / 1000) * sampling_rate)
#
#     Gain_Log_Mel_spectrogram(signal=signal_final, sr=sampling_rate, n_fft=512, n_mels=40,
#                              window="ham", win_length=window_length, hop_length=window_shift,
#                              save_folder="Spectrograms_utterance", session=session_num, emotion=emotion_class,
#                              count=count2)


