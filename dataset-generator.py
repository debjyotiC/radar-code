import pandas as pd
import numpy as np
from os import listdir
from os.path import isdir, join

dataset_path = 'data/csv_files/home_new'

# Config parameters for test
configParameters = {'numDopplerBins': 16, 'numRangeBins': 128, 'rangeResolutionMeters': 0.04360212053571429,
                    'rangeIdxToMeters': 0.04360212053571429, 'dopplerResolutionMps': 0.12518841691334906,
                    'maxRange': 10.045928571428572, 'maxVelocity': 2.003014670613585} # AWR2944X_Deb

# configParameters = {'numDopplerBins': 16, 'numRangeBins': 256, 'rangeResolutionMeters': 0.146484375,
#                     'rangeIdxToMeters': 0.146484375, 'dopplerResolutionMps': 0.1252347734553042, 'maxRange': 33.75,
#                     'maxVelocity': 0.5009390938212168}  # xwr16xx_umbc
# configParameters = {'numDopplerBins': 16, 'numRangeBins': 256, 'rangeResolutionMeters': 0.146484375,
#                     'rangeIdxToMeters': 0.146484375, 'dopplerResolutionMps': 0.1252347734553042, 'maxRange': 33.75,
#                     'maxVelocity': 0.5009390938212168}  # xwr16xx_umbc_indoor

# configParameters = {'numDopplerBins': 16, 'numRangeBins': 256, 'rangeResolutionMeters': 0.146484375,
#                     'rangeIdxToMeters': 0.146484375, 'dopplerResolutionMps': 0.1252347734553042, 'maxRange': 33.75,
#                     'maxVelocity': 1.0018781876424336}  # xwr16xx_umbc_outdoor

all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
# all_targets.remove('.DS_Store')

print(all_targets)

filenames = []
y = []

for index, target in enumerate(all_targets):
    filenames.append(listdir(join(dataset_path, target)))
    y.append(np.ones(len(filenames[index])) * index)


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


def calc_range_doppler(data_frame, packet_id, config):
    payload = data_frame[packet_id].to_numpy()
    # Convert levels to dBm
    payload = 20 * np.log10(payload)
    # Clac. range Doppler array
    rangeDoppler = np.reshape(payload, (config["numDopplerBins"], config["numRangeBins"]), 'F')
    rangeDoppler = np.append(rangeDoppler[int(len(rangeDoppler) / 2):], rangeDoppler[:int(len(rangeDoppler) / 2)],
                             axis=0)

    return rangeDoppler


out_x_range_doppler = []
out_x_range_doppler_cfar = []
out_y_range_doppler = []

for folder in range(len(all_targets)):
    all_files = join(dataset_path, all_targets[folder])
    for i in range(len(listdir(all_files))):
        full_path = join(all_files, listdir(all_files)[i])

        print(full_path, folder)

        df_data = pd.read_csv(full_path)

        for col in df_data.columns:
            data = calc_range_doppler(df_data, col, configParameters)
            cfar_data = apply_2d_cfar(data, guard_band_width=3, kernel_size=3, threshold_factor=1)

            out_x_range_doppler.append(data)
            out_x_range_doppler_cfar.append(cfar_data)
            out_y_range_doppler.append(folder + 1)

data_range_x = np.array(out_x_range_doppler)
data_range_cfar_x = np.array(out_x_range_doppler_cfar)
data_range_y = np.array(out_y_range_doppler)

np.savez('data/npz_files/range_doppler_home_data_test.npz', out_x=data_range_x, out_y=data_range_y)
np.savez('data/npz_files/range_doppler_home_cfar_data_test.npz', out_x=data_range_cfar_x, out_y=data_range_y)
