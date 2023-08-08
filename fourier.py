import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fft


plt.rcParams["savefig.dpi"] = 400
plt.style.use("bmh")

# read monthly data and convert to numpy array
data = pd.read_csv("Sunspots_observations.csv", skiprows=6)
data = data.loc[data["Year CE.1"].notnull(), ["Year CE.1", "SNm"]].to_numpy()

# t: time in years, y: monthly mean total sunspot number
t = data[:, 0]
y = data[:, 1]

# plot time series (zero mean)
plt.figure(figsize=(12, 4))
plt.plot(t, y - y.mean())
plt.xlabel("Time")
plt.ylabel("Normalized count")
plt.title("Monthly mean total sunspot number by year")
plt.tight_layout()

n_samples = len(y)
current_y = y - y.mean()  # normalize

# approximate observations as a sum of n cosines using fourier analysis
n = 15
plot_steps = False  # show evolution at each step

estimates_A = np.zeros(n)  # magnitudes
estimates_T = np.zeros(n)  # periods
estimates_P = np.zeros(n)  # phases
print("-"*50)

for iter in range(n):
    print(f"Iteration {iter+1}")

    # apply the FFT on the signal
    fourier = fft.rfft(current_y)
    # normalize the FFT output
    amplitudes = 2 / n_samples * np.abs(fourier)
    # get the frequency components of the spectrum
    timestep = (t.max() - t.min()) / (n_samples - 1)  # sample spacing
    frequencies = fft.rfftfreq(n_samples, d=timestep)

    # index of peak amplitude
    index = np.argmax(amplitudes)
    # strongest frequency component
    frequency = frequencies[index]
    period = 1 / frequency
    print(f"Period: {period}")
    # normalized amplitude
    magnitude = 2 / n_samples * np.abs(fourier[index])
    print(f"Magnitude: {magnitude}")
    # phase, with time shift 
    phase = (np.angle(fourier[index]) - 2 * np.pi / period * t.min()) % (2 * np.pi) 
    print(f"Phase: {phase}")
    # or represent as cosine and sine, without phase
    B1 = magnitude * np.cos(phase)  # cosine multiplier
    B2 = - magnitude * np.sin(phase)  # sine multiplier
    print(f"B1: {B1}, B2: {B2}")

    estimates_A[iter] = magnitude
    estimates_T[iter] = period
    estimates_P[iter] = phase

    if plot_steps:
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        # plot the normalized FFT
        plt.stem(frequencies, amplitudes, markerfmt=" ")
        plt.xscale("log")
        plt.xlabel("Frequency (log)")
        plt.ylabel("Amplitude")
        plt.title("Normalized FFT spectrum")

        plt.subplot(3, 1, 2)
        plt.plot(t, current_y, label="Current signal")
        plt.plot(t, magnitude * np.cos(2 * np.pi / period * t + phase), label="Estimate")
        #plt.plot(t, B1 * np.cos(2 * np.pi / period * t) + B2 * np.sin(2 * np.pi / period * t), c="orange")
        plt.xlabel("Time")
        plt.ylabel("Normalized count")
        plt.title("Estimated component")
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(t, current_y - magnitude * np.cos(2 * np.pi / period * t + phase))
        #plt.plot(t, current_y - B1 * np.cos(2 * np.pi / period * t) - B2 * np.sin(2 * np.pi / period * t), c="orange")
        plt.xlabel("Time")
        plt.ylabel("Normalized count")
        plt.title("Residuals")

        plt.suptitle(f"Iteration {iter+1}")
        plt.tight_layout()
    
    print("-"*50)
    # remove estimated component from the signal
    current_y = current_y - magnitude * np.cos(2 * np.pi / period * t + phase)

# assume gaussian noise
sigma = np.std(current_y)
print(f"Sigma (SD of residuals): {sigma}")

# plot time series of the residuals
plt.figure(figsize=(12, 4))
plt.plot(t, current_y)
plt.xlabel("Time")
plt.ylabel("Normalized count")
plt.title(f"Residuals after iteration {iter+1}")
plt.tight_layout()

# plot distribution of residuals
plt.figure(figsize=(6, 4))
plt.hist(current_y, bins=100)
plt.ylabel("Count")
plt.title(f"Distribution of residuals after iteration {iter+1}")
plt.tight_layout()

# plot reconstructed signal as sum of n cosines
signal = np.zeros(len(t))
for magnitude, period, phase in zip(estimates_A, estimates_T, estimates_P):
    signal += magnitude * np.cos(2 * np.pi / period * t + phase)
plt.figure(figsize=(12, 4))
plt.plot(t, y - y.mean(), label="Original")
plt.plot(t, signal, label="Reconstructed")
plt.xlabel("Time")
plt.ylabel("Normalized count")
plt.title("Reconstructed signal")
plt.legend()
plt.tight_layout()

# save to disk the results of fourier analysis under their corresponding name
np.savez("fourier.npz", estimates_A=estimates_A, estimates_T=estimates_T, estimates_P=estimates_P, sigma=[sigma])

plt.show()