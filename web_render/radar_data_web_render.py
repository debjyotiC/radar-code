from flask import Flask, render_template
import real_time_classifier
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model_path = "saved-tflite-model/range-doppler-float16.tflite"


def apply_2d_cfar(signal, guard_band_width, kernel_size, threshold_factor):
    num_rows, num_cols = signal.shape
    thresholded_signal = np.zeros((num_rows, num_cols))
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
                thresholded_signal[i, j] = 1
    return thresholded_signal


def classify_data(data):
    # 2D CFAR parameters
    guard_band_width = 3
    kernel_size = 3
    threshold_factor = 1
    range_doppler_cfar = apply_2d_cfar(data, guard_band_width, kernel_size, threshold_factor)[:, :128]

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_index = input_details["index"]

    in_tensor = np.float32(range_doppler_cfar.reshape(1, range_doppler_cfar.shape[0], range_doppler_cfar.shape[1], 1))
    interpreter.set_tensor(input_index, in_tensor)
    interpreter.invoke()
    classes = interpreter.get_tensor(output_details['index'])[0]
    pred = np.argmax(classes)

    return pred


@app.route("/")
def index():
    # get results
    doppler_data = real_time_classifier.range_doppler_arr
    prediction = classify_data(doppler_data)
    classes_values = ["occupied_room", "empty_room"]

    # render the HTML template with results
    return render_template("index.html", results=classes_values[prediction])


if __name__ == '__main__':
    app.run(debug=True)
