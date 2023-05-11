import numpy as np
import matplotlib.pyplot as plt

moving = 1
no_moving = 2

frange_doppler_features = np.load("data/npz_files/range_doppler_home_data_test.npz", allow_pickle=True)

x_data, y_data = frange_doppler_features['out_x'], frange_doppler_features['out_y']
# Config parameters for test
configParameters = {'numDopplerBins': 16, 'numRangeBins': 128, 'rangeResolutionMeters': 0.04360212053571429,
                    'rangeIdxToMeters': 0.04360212053571429, 'dopplerResolutionMps': 0.12518841691334906,
                    'maxRange': 10.045928571428572, 'maxVelocity': 2.003014670613585}  # AWR2944X_Deb

# configParameters = {'numDopplerBins': 16, 'numRangeBins': 256, 'rangeResolutionMeters': 0.146484375,
#                     'rangeIdxToMeters': 0.146484375, 'dopplerResolutionMps': 0.1252347734553042, 'maxRange': 33.75,
#                     'maxVelocity': 0.5009390938212168}  # xwr16xx_umbc
# configParameters = {'numDopplerBins': 16, 'numRangeBins': 256, 'rangeResolutionMeters': 0.146484375,
#                     'rangeIdxToMeters': 0.146484375, 'dopplerResolutionMps': 0.1252347734553042, 'maxRange': 33.75,
#                     'maxVelocity': 0.5009390938212168} # xwr16xx_umbc_indoor

# configParameters = {'numDopplerBins': 16, 'numRangeBins': 256, 'rangeResolutionMeters': 0.146484375,
#                     'rangeIdxToMeters': 0.146484375, 'dopplerResolutionMps': 0.1252347734553042, 'maxRange': 33.75,
#                     'maxVelocity': 1.0018781876424336}  # xwr16xx_umbc_outdoor

# Generate the range and doppler arrays for the plot
rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
dopplerArray = np.multiply(np.arange(-configParameters["numDopplerBins"] / 2, configParameters["numDopplerBins"] / 2),
                           configParameters["dopplerResolutionMps"])

fig = plt.figure()

test = moving  # change this for testing

for count, frame in enumerate(x_data[np.where(y_data == test)]):
    plt.clf()
    plt.xlabel("Range (m)")
    plt.ylabel("Doppler velocity (m/s)")
    if test - 1:
        plt.title(f"Frame {count} for no moving target/empty area")
    else:
        plt.title(f"Frame {count} for moving target")
    cs = plt.contourf(rangeArray, dopplerArray, frame)
    print(rangeArray.shape)
    print(dopplerArray.shape)
    print(frame.shape)
    fig.colorbar(cs, shrink=0.9)
    fig.canvas.draw()
    plt.pause(0.1)
