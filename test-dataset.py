import numpy as np
import pywt
from scipy.fft import fft
import matplotlib.pyplot as plt

moving = 1
no_moving = 2

frange_doppler_features = np.load("data/npz_files/range_doppler_home_data.npz", allow_pickle=True)

x_data, y_data = frange_doppler_features['out_x'], frange_doppler_features['out_y']
# Config parameters for test
configParameters = {'numDopplerBins': 16, 'numRangeBins': 128, 'rangeResolutionMeters': 0.04360212053571429,
                    'rangeIdxToMeters': 0.04360212053571429, 'dopplerResolutionMps': 0.12518841691334906,
                    'maxRange': 10.045928571428572, 'maxVelocity': 2.003014670613585}  # AWR2944X_Deb

# configParameters = {'numDopplerBins': 16, 'numRangeBins': 256, 'rangeResolutionMeters': 0.146484375,
#                     'rangeIdxToMeters': 0.146484375, 'dopplerResolutionMps': 0.1252347734553042, 'maxRange': 33.75,
#                     'maxVelocity': 1.0018781876424336}

# Generate the range and doppler arrays for the plot
rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
dopplerArray = np.multiply(np.arange(-configParameters["numDopplerBins"] / 2, configParameters["numDopplerBins"] / 2),
                           configParameters["dopplerResolutionMps"])

fig = plt.figure()

test = moving  # change this for testing


def wavelet_denoising(data, wavelet, value):
    # Perform the wavelet transform.
    coefficients = pywt.wavedec2(data, wavelet)

    # Threshold the coefficients.
    threshold = pywt.threshold(coefficients[0], value=value)
    coefficients[0] = pywt.threshold(coefficients[0], threshold)

    # Inverse wavelet transform.
    denoised_data = pywt.waverec2(coefficients, wavelet)

    return denoised_data


def apply_2d_cfar(signal, guard_band_width, kernel_size, threshold_factor):
    num_rows, num_cols = signal.shape
    threshold_signal = np.zeros((num_rows, num_cols))
    for i in range(guard_band_width, num_rows - guard_band_width):
        for j in range(guard_band_width, num_cols - guard_band_width):
            # Estimate the noise level
            noise_level = np.mean(np.concatenate((
                signal[i - guard_band_width:i + guard_band_width, j - guard_band_width:j + guard_band_width].ravel(),
                signal[i - kernel_size:i + kernel_size, j - kernel_size:j + kernel_size].ravel())))
            # Calculate the threshold for detection
            threshold = threshold_factor * noise_level
            # Check if the signal exceeds the threshold
            if signal[i, j] > threshold:
                threshold_signal[i, j] = 1
    return threshold_signal


def calculate_distance(range_doppler_data):
    num_doppler_bins = range_doppler_data.shape[0]
    distances = []

    for doppler_bin_index in range(num_doppler_bins):
        doppler_profile = range_doppler_data[doppler_bin_index, :]

        # Perform Range FFT
        doppler_profile_fft = fft(doppler_profile)

        # Find peak frequency (Doppler shift)
        peak_bin_index = np.argmax(np.abs(doppler_profile_fft))

        # Calculate peak frequency index based on the sampling rate
        peak_frequency_index = peak_bin_index - num_doppler_bins // 2

        # Calculate peak frequency in m/s
        peak_frequency = peak_frequency_index * configParameters['dopplerResolutionMps']

        # Calculate distance using range resolution and range index to meters conversion factor
        distance = peak_frequency * configParameters['rangeResolutionMeters'] / 2.0
        distances.append(distance)

    average_distance = np.mean(distances)
    return average_distance


for count, frame in enumerate(x_data[np.where(y_data == test)]):
    dist = calculate_distance(frame)
    plt.clf()
    plt.xlabel("Range (m)")
    plt.ylabel("Doppler velocity (m/s)")
    if test - 1:
        plt.title(f"Frame {count} for no moving target/empty area {dist}")
    else:
        plt.title(f"Frame {count} for moving target {dist}")

    # frame = wavelet_denoising(frame,  wavelet='haar', value=2.0)
    # frame = apply_2d_cfar(frame, guard_band_width=5, kernel_size=5, threshold_factor=2.2)
    cs = plt.contourf(rangeArray, dopplerArray, frame)
    fig.colorbar(cs, shrink=0.9)
    fig.canvas.draw()
    plt.pause(0.1)
