import numpy as np
import tensorflow as tf
import time

type_of_quantization = "float16"
model_path = f"saved-tflite-model/range-doppler-home-{type_of_quantization}.tflite"

range_doppler_features = np.load("data/npz_files/range_doppler_home_cfar_data.npz", allow_pickle=True)
x_data, y_data = range_doppler_features['out_x'], range_doppler_features['out_y']

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_index = input_details["index"]
classes_values = ["occupied_room", "empty_room"]

print(f"Model type {type_of_quantization}")
# bad testing on training data
for i, true_label in enumerate(y_data):
    data = x_data[i]

    in_tensor = np.float32(data.reshape(1, data.shape[0], data.shape[1], 1))
    # print(in_tensor.shape)
    start_time = time.time()
    interpreter.set_tensor(input_index, in_tensor)
    interpreter.invoke()
    classes = interpreter.get_tensor(output_details['index'])[0]
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000.0

    pred = np.argmax(classes)
    print(f"Inference time: {elapsed_time} ms")
    print(classes_values[pred], classes_values[true_label-1])
