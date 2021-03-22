import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd

#%%
#Applying Empirical Model Decomposition manually
# 1. Compute all local extrema in the signal
def Compute_local_extrema(signal):
    maximas = []
    t_maximas = []

    minimas = []
    t_minimas = []

    for i in range(1, len(signal)-1):
        #Check and restore the maximas
        if (signal[i] >= signal[i-1]) and (signal[i] >= signal[i+1]):
            maximas.append(signal[i])
            t_maximas.append(i)
        #Check and restore the minimas
        elif (signal[i] <= signal[i-1]) and (signal[i] <= signal[i+1]):
            minimas.append(signal[i])
            t_minimas.append(i)

    return maximas,t_maximas,minimas,t_minimas


#2. Construct the upper, lower envelope Cubic-Splines with regard to the extrema points
#3. Compute the mean envelop CS based on upper and lower ones
def Compute_envelope_cubic_splines(signal, maximas, t_maximas, minimas, t_minimas):
    cs_upper_points = CubicSpline(t_maximas, maximas)
    time_range = np.array(range(len(signal)))

    cs_lower_points = CubicSpline(t_minimas, minimas)

        #Calculate the whole upper and lower envelope Cubic-Splines across the entire time range
    cs_upper = cs_upper_points(time_range)
    cs_lower = cs_lower_points(time_range)


    #Calculate the mean of the upper and lower envelop Cubic-Splines
    cs_mean = (cs_upper+cs_lower)/2

        #Plot the upper, lower, and the mean of the envelope Cubic-Splines
    # plt.plot(time_range, cs_upper, label='upper spline', color='g')
    # plt.plot(time_range, cs_lower, label='lower spline', color='y')
    # plt.plot(time_range, cs_mean, label='mean spline', color='r')
    # plt.legend(loc='lower right', ncol=2)
    # plt.title("mean cubic spline")
    # plt.show()

    return cs_upper, cs_lower, cs_mean


# 4. Subtract the mean from the original signal, then get a new data sequence
def Decompose_signal(signal, cs_mean):

    sequence = signal - cs_mean
    time_range = np.array(range(len(signal)))

        #Plot the signal, substraction of the mean, and the result sequence
    # plt.plot(time_range, signal, label='original signal')
    # plt.plot(time_range, cs_mean, label='mean spline')
    plt.plot(time_range, sequence, label='new sequence')
    plt.title("Signal removed mean cubic spline")
    plt.legend(loc='upper right', ncol=2)
    plt.show()

    return sequence


#5. Reapt 1~4 steps, until the result sequence satisfy the IMF criteria
#5.1 Set up the criteria for checking the result sequence satisfies the IMF criteria
def check_if_IMF(sequence):
    #The number of extrema <= the number of Zero Crossing
        #Calculate the number of Zero Crossing of sequence
    z_index = librosa.zero_crossings(sequence, zero_pos=True, pad=False)

    n_zc = sum(z_index)

        #Calculate the number of local extrema in the sequence
    maximas, t_maximas, minimas, t_minimas = Compute_local_extrema(sequence)
    n_extrema = len(maximas)+len(minimas)

    #The mean of the envelopes must be zero at all times
    cs_upper, cs_lower, cs_mean = Compute_envelope_cubic_splines(sequence, maximas, t_maximas, minimas, t_minimas)

    #Return whether the criteria are sastisfied
    return (n_extrema <= n_zc) and (all(x==0 for x in cs_mean))


#5.2 Check the criteria and repeat the 1~4 steps if they are not satisfied
def Repeat_to_get_IMF(signal, iteration_max):
    iteration=0

    sequence = signal

    while True:

        maximas_seq, t_maximas_seg, minimas_seg, t_minimas_seg = Compute_local_extrema(signal=sequence)
        cs_upper_seq, cs_lower_seq, cs_mean_seq = Compute_envelope_cubic_splines(sequence, maximas_seq, t_maximas_seg, minimas_seg, t_minimas_seg)
        sequence = Decompose_signal(sequence, cs_mean_seq)

        iteration+=1

        if (iteration<=iteration_max) and (check_if_IMF(sequence)):
            IMF = sequence
            return IMF
        elif (iteration<=iteration_max) and not (check_if_IMF(sequence)):
            continue
        else:
            return False


#6. Subtract this IMF from the original signal
#7. Repeat steps 1–6 till there are no more IMFs left in the residual signal to derive all the IMFs in the signal
def Check_imf_in_residual_to_get_all_IMFs(signal):
    signal_residual = signal

    IMFs=[]
    while True:
        imf = Repeat_to_get_IMF(signal_residual, iteration_max=10)

        if not imf:
            break
        else:
            IMFs.append(imf)
            signal_residual = signal - imf

    return IMFs, signal_residual

#%% Visualise some of the results mentioned in above steps
file = "/Users/talen/Documents/Datasets/IEMOCAP/Data/Ses01F_impro01_F000.wav"
audio, sr = librosa.load(file, sr=16000)

maximas,t_maximas,minimas,t_minimas = Compute_local_extrema(signal=audio)
    #Visualise the orignial signal
# librosa.display.waveplot(y=audio, sr=sr)
# plt.title("The raw audio signal")
# plt.show()
    #Visualise the cubic splines of the upper, lower, and mean evelope
cs_upper, cs_lower, cs_mean = Compute_envelope_cubic_splines(signal=audio, maximas=maximas,t_maximas=t_maximas,minimas=minimas,
                               t_minimas=t_minimas)

    #Visualise the signal after removal of the mean spline
Decompose_signal(signal=audio, cs_mean=cs_mean)