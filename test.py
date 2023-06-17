import numpy as np

range_doppler_features = np.load("data/npz_files/range_doppler_umbc_new_2_data.npz", allow_pickle=True)

x_data, y_data = range_doppler_features['out_x'], range_doppler_features['out_y']

matrix = x_data[0]

max_value = np.max(matrix)
max_indices = np.unravel_index(np.argmax(matrix), matrix.shape)

# Print the maximum value and its indices
print("Maximum value:", max_value)
print("Indices of the maximum value:", max_indices)

configParameters = {'numDopplerBins': 16, 'numRangeBins': 256, 'rangeResolutionMeters': 0.146484375,
                    'rangeIdxToMeters': 0.146484375, 'dopplerResolutionMps': 0.1252347734553042, 'maxRange': 33.75,
                    'maxVelocity': 1.0018781876424336}

# Generate the range and doppler arrays for the plot
rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
dopplerArray = np.multiply(np.arange(-configParameters["numDopplerBins"] / 2, configParameters["numDopplerBins"] / 2),
                           configParameters["dopplerResolutionMps"])

print(rangeArray[24])
