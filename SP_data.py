import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import pywt
import scipy
import time

FIG_SIZE = (15,10)


file = "spkr09_M_S1a_cry14.wav"
##for file in range(0, len(audio_file),1)

# load audio file with Librosa
start = time.clock()
signal, sample_rate = librosa.load(file, sr=48000)
print(signal.shape, signal)
print(sample_rate)
print(time.clock()-start) #just to check the full time loading 

#librosa.time_to_frames(np.arange(0, 1, 0.1), sr=48000, hop_length=512)

# WAVEFORM
# display waveform
plt.figure(figsize=FIG_SIZE)
librosa.display.waveplot(signal, sample_rate, alpha=0.4)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.savefig('wavfile.png')  



# FFT -> power spectrum
# perform Fourier transform (N=fs*t -> N=num of samples)
fs=48000
fft = np.fft.fft(signal)
t= np.linspace(1/fs, len(signal)/fs, len(signal))
#t = np.arange(0,1, float(1/fs))
freq = np.fft.fftfreq(len(fft), t[1] - t[0])
#plot ftt
plt.figure(figsize=FIG_SIZE)
plt.plot(freq, np.abs(fft), 'g--',marker='o')
plt.xlabel("Frequency")
plt.title("np=x-point FFT")
plt.savefig('fft.png')  

# calculate abs values on complex numbers to get magnitude
spectrum = np.abs(fft)

# create frequency variable
f = np.linspace(0, sample_rate, len(spectrum))

# take half of the spectrum and frequency
left_spectrum = spectrum[:int(len(spectrum)/2)]
left_f = f[:int(len(spectrum)/2)]

# plot spectrum
plt.figure(figsize=FIG_SIZE)
plt.plot(left_f, left_spectrum, alpha=0.4)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")
plt.savefig('spectrum.png') 
plt.show()



# STFT -> spectrogram
hop_length = 512 # in num. of samples / stride
n_fft = 2048 # window in num. of samples / window_size
win_length = n_fft
#ham_win = np.hamming(2048)


# calculate duration hop length and window in seconds
hop_length_duration = float(hop_length)/sample_rate
n_fft_duration = float(n_fft)/sample_rate

print("STFT hop length duration is: {}s".format(hop_length_duration))
print("STFT window duration is: {}s".format(n_fft_duration))

# perform stft
stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length,win_length=n_fft,window='hamming')

# calculate abs values on complex numbers to get magnitude
spectrogram = np.abs(stft)

# display spectrogram
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length, y_axis='log', x_axis='time')
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.title("Spectrogram")
plt.savefig('spectrogram.png') 

# apply logarithm to cast amplitude to Decibels
log_spectrogram = librosa.amplitude_to_db(spectrogram)

plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length, y_axis='log', x_axis='time')
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")
plt.savefig('spectrogram(dB).png') 
plt.show()


#cwt dwt 
continuous_wavelet = pywt.ContinuousWavelet('mexh')
print(continuous_wavelet)

max_scale = 20
scales = np.arange(1, max_scale + 1)
cwtmatr, freqs = pywt.cwt(signal, scales, continuous_wavelet, 48000)

# visualize cwt
plt.figure(figsize=(4,4))
plt.subplot(3,1,1)
(phi, psi) = continuous_wavelet.wavefun()
plt.plot(psi,phi)
plt.savefig('mexh.png')  

plt.grid()
plt.figure(figsize=(15,10))
plt.subplot(3,1,2)
plt.title('cwt')
plt.xlabel('Samples')
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.plot(np.linspace(0.0, len(signal),len(signal)), signal)
plt.xlim(xmin=0)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.savefig('cwt2.png')  
plt.grid()
plt.figure(figsize=(15,10))
plt.subplot(3,1,3)
plt.imshow(cwtmatr, extent=[0, int(len(signal)), 1, max_scale + 1],cmap='PRGn', aspect='auto', 
vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.savefig('cwt2.png')  

discrete_wavelet = pywt.Wavelet('db2')
print(discrete_wavelet)

max_level = pywt.dwt_max_level(len(signal), discrete_wavelet)
print('MAXIMUM DECOMPOSE LEVEL = ',max_level)

# decompose
tree = pywt.wavedec(signal, 'db2',level=3)
cA3, cD3, cD2, cD1 = tree
#print(len(cD1),len(cD2),len(cD3),len(cA3))

# reconstruct
rec_sample = pywt.waverec(tree, 'db2')
rec_to_orig = pywt.idwt(None, cD1, 'db2', 'smooth')  #
rec_to_level1 = pywt.idwt(None, cD2, 'db2', 'smooth')
rec_to_level2_from_detail = pywt.idwt(None, cD3, 'db2', 'smooth')
rec_to_level2_from_approx = pywt.idwt(cA3, None, 'db2', 'smooth')
#print(len(rec_to_orig),len(rec_to_level1),len(rec_to_level2_from_detail),len(rec_to_level2_from_approx))

# visualize dwt
plt.figure(figsize=(4,4))
(phi, psi, x) = discrete_wavelet.wavefun()
plt.plot(x, phi)
plt.savefig('dwt2.png')  

plt.grid()
plt.figure(figsize=(15,10))
plt.subplot(5,1,1)
plt.title('dwt2')
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.xlabel('Samples')
plt.plot(np.linspace(0.0, len(signal),len(signal)), signal)
plt.xlim(xmin=0)
plt.grid()

plt.subplot(5,1,2)
plt.title('cD1')
plt.plot(np.linspace(0.0, len(rec_to_orig),len(rec_to_orig)), rec_to_orig)
plt.xlim(xmin=0)
plt.grid()
plt.subplot(5,1,3)
plt.title('cD2')
plt.plot(np.linspace(0.0, len(rec_to_level1),len(rec_to_level1)), rec_to_level1)
plt.xlim(xmin=0)
plt.grid()
plt.subplot(5,1,4)
plt.title('cD3')
plt.plot(np.linspace(0.0, len(rec_to_level2_from_detail),len(rec_to_level2_from_detail)), rec_to_level2_from_detail)
plt.xlim(xmin=0)
plt.grid()
plt.subplot(5,1,5)
plt.title('cA3')
plt.plot(np.linspace(0.0, len(rec_to_level2_from_approx),len(rec_to_level2_from_approx)), rec_to_level2_from_approx)
plt.xlim(xmin=0)
plt.grid()
plt.tight_layout()
plt.savefig('dwt2.png')  
#LP linear predicition 
#a = librosa.lpc(signal, 2)
#signal_hat = scipy.signal.lfilter([0] + -1*a[1:], [1], signal)
#plt.figure()
#plt.plot(signal)
#plt.plot(signal_hat, linestyle='--')
#plt.legend(['signal', 'signal_hat'])
#plt.title('LP Model Forward Prediction')
#plt.savefig('lp.png')  

# MFCCs
# extract 13 MFCCs
MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=10)

# display MFCCs
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("MFCCs")
plt.savefig('MFCC.png')  

# show plots
plt.show()
