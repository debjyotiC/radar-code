import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

type_of_quantization = "float16"
model_path = f"saved-tflite-model/range-doppler-home-{type_of_quantization}.tflite"

range_doppler_features = np.load("data/npz_files/range_doppler_home_cfar_data_test.npz", allow_pickle=True)
x_data, y_data = range_doppler_features['out_x'], range_doppler_features['out_y']

range_doppler_features_raw = np.load("data/npz_files/range_doppler_home_data_test.npz", allow_pickle=True)
x_data_raw, y_data_raw = range_doppler_features_raw['out_x'], range_doppler_features['out_y']

# Randomise data
# Get the number of examples
num_examples = x_data.shape[0]
perm = np.random.permutation(num_examples)
x_data, y_data = x_data[perm], y_data[perm]
x_data_raw, y_data_raw = x_data_raw[perm], y_data_raw[perm]

# Config parameters for test
configParameters = {'numDopplerBins': 16, 'numRangeBins': 128, 'rangeResolutionMeters': 0.04360212053571429,
                    'rangeIdxToMeters': 0.04360212053571429, 'dopplerResolutionMps': 0.12518841691334906,
                    'maxRange': 10.045928571428572, 'maxVelocity': 2.003014670613585}  # AWR2944X_Deb
rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
dopplerArray = np.multiply(np.arange(-configParameters["numDopplerBins"] / 2, configParameters["numDopplerBins"] / 2),
                           configParameters["dopplerResolutionMps"])
# Load the TensorFlow Lite model.
interpreter = tf.lite.Interpreter(model_path=model_path)

interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_index = input_details["index"]
output_index = output_details['index']


# Define a function to compute the CAM.
def compute_cam(tflite_interpreter, mat, layer_name, index_input, index_output):
    # Get the index of the last convolutional layer.
    last_conv_index = 0

    for i in range(len(tflite_interpreter.get_output_details())):
        if tflite_interpreter.get_output_details()[i]['name'] == layer_name:
            last_conv_index = i
            break

    # Set the input tensor to the input image.
    tflite_interpreter.set_tensor(index_input, mat)

    # Run the model and get the output of the last convolutional layer and the softmax predictions.
    tflite_interpreter.invoke()
    last_conv_output, predictions = tflite_interpreter.get_tensor(last_conv_index), tflite_interpreter.get_tensor(index_output)

    # Compute the CAM by taking a weighted sum of the feature maps in the last convolutional layer.
    cam = np.sum(last_conv_output * np.expand_dims(predictions, axis=1), axis=-1)
    cam = ((cam - np.min(cam)) / (np.max(cam) - np.min(cam)))[0]
    # cam = np.uint8(255 * cam)[0]
    return cam


fig = plt.figure()

for count, frame in enumerate(x_data):
    in_tensor = np.float32(frame.reshape(1, frame.shape[0], frame.shape[1], 1))
    cam_mat = compute_cam(interpreter, in_tensor, 'conv2d_2', input_index, output_index)
    # Convert the NumPy array to a PIL image
    cam_mat = (cam_mat * 255).astype(np.uint8)
    image = Image.fromarray(cam_mat)

    # Save the PIL image
    save_as = 'saved_image.png'
    image.save(save_as)

    # print(cam_mat.shape)
    # pred = interpreter.get_tensor(output_details['index'])[0]
    #
    # classes = np.argmax(pred)
    # conf_score = round(pred[0], 2)
    # # Pretty print to plot
    # plt.clf()
    # plt.xlabel("Range (m)")
    # plt.ylabel("Doppler velocity (m/s)")
    # cs_1 = plt.contourf(rangeArray, dopplerArray, x_data_raw[count])
    #
    # if not classes:
    #     plt.title(f"Frame {count} for moving target {conf_score}")
    #     cs_2 = plt.contour(rangeArray, dopplerArray, cam_mat, levels=1, colors='r')
    # else:
    #     plt.title(f"Frame {count} for empty area {conf_score}")
    # fig.colorbar(cs_1, shrink=0.9)
    # legend_labels = [
    #     Line2D([0], [0], color='r', linestyle='-')
    # ]
    # legend_text = [f"Red: Moving target"]
    # # Add a legend
    # plt.legend(legend_labels, legend_text)
    # fig.canvas.draw()
    # plt.pause(0.2)
