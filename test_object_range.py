import numpy as np

frange_doppler_features = np.load("data/npz_files/range_doppler_home_data.npz", allow_pickle=True)

x_data, y_data = frange_doppler_features['out_x'], frange_doppler_features['out_y']

configParameters = {'numDopplerBins': 16, 'numRangeBins': 128, 'rangeResolutionMeters': 0.04360212053571429,
                    'rangeIdxToMeters': 0.04360212053571429, 'dopplerResolutionMps': 0.12518841691334906,
                    'maxRange': 10.045928571428572, 'maxVelocity': 2.003014670613585}  # AWR2944X_Deb

# Generate the range and doppler arrays for the plot
rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
dopplerArray = np.multiply(np.arange(-configParameters["numDopplerBins"] / 2, configParameters["numDopplerBins"] / 2),
                           configParameters["dopplerResolutionMps"])


def highlight_peaks(matrix, thres):
    rows, cols = matrix.shape
    peaks = []

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if matrix[i, j] >= thres:
                neighbors = matrix[i - 1:i + 2, j - 1:j + 2]
                if matrix[i, j] == np.max(neighbors):
                    peaks.append((i, j))

    return peaks


cfar_matrix = x_data[0]
threshold = 70.0

highlighted_peaks = highlight_peaks(cfar_matrix, threshold)

highlighted_peaks_array = np.array(highlighted_peaks)
result = rangeArray[highlighted_peaks_array[:, 1]].round(2)
print(result)
# print("Highlighted peaks with velocity:")
# print(highlighted_peaks)
# print(rangeArray)


# elements = [0.0, 0.04360212, 0.08720424, 0.13080636, 0.17440848, 0.2180106,
#             0.26161272, 0.30521484, 0.34881696, 0.39241908, 0.43602121, 0.47962333,
#             0.52322545, 0.56682757, 0.61042969, 0.65403181, 0.69763393, 0.74123605,
#             0.78483817, 0.82844029, 0.87204241, 0.91564453, 0.95924665, 1.00284877,
#             1.04645089, 1.09005301, 1.13365513, 1.17725725, 1.22085938, 1.2644615,
#             1.30806362, 1.35166574, 1.39526786, 1.43886998, 1.4824721, 1.52607422,
#             1.56967634, 1.61327846, 1.65688058, 1.7004827, 1.74408482, 1.78768694,
#             1.83128906, 1.87489118, 1.9184933, 1.96209542, 2.00569754, 2.04929967,
#             2.09290179, 2.13650391, 2.18010603, 2.22370815, 2.26731027, 2.31091239,
#             2.35451451, 2.39811663, 2.44171875, 2.48532087, 2.52892299, 2.57252511,
#             2.61612723, 2.65972935, 2.70333147, 2.74693359, 2.79053571, 2.83413783,
#             2.87773996, 2.92134208, 2.9649442, 3.00854632, 3.05214844, 3.09575056,
#             3.13935268, 3.1829548, 3.22655692, 3.27015904, 3.31376116, 3.35736328,
#             3.4009654, 3.44456752, 3.48816964, 3.53177176, 3.57537388, 3.618976,
#             3.66257813, 3.70618025, 3.74978237, 3.79338449, 3.83698661, 3.88058873,
#             3.92419085, 3.96779297, 4.01139509, 4.05499721, 4.09859933, 4.14220145,
#             4.18580357, 4.22940569, 4.27300781, 4.31660993, 4.36021205, 4.40381417,
#             4.44741629, 4.49101842, 4.53462054, 4.57822266, 4.62182478, 4.6654269,
#             4.70902902, 4.75263114, 4.79623326, 4.83983538, 4.8834375, 4.92703962,
#             4.97064174, 5.01424386, 5.05784598, 5.1014481, 5.14505022, 5.18865234,
#             5.23225446, 5.27585658, 5.31945871, 5.36306083, 5.40666295, 5.45026507,
#             5.49386719, 5.53746931]
# #
# indices = [(8, 3), (8, 30), (8, 32), (8, 38), (8, 42), (8, 49), (8, 52), (8, 59),
#            (8, 63), (8, 65), (8, 70), (8, 74), (8, 77), (8, 80), (8, 82), (8, 85),
#            (8, 88), (8, 92), (8, 95), (8, 100), (8, 104), (8, 113)]
#
# picked_elements = [elements[idx[1]] for idx in indices]
# print(picked_elements)
