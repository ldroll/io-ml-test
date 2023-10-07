from tflite_runtime.interpreter import Interpreter
import numpy as np
from PIL import Image
import pathlib

# Input Data Configuration
#data_dir = "/home/pi/io-ml-test/"
model_path = "saved-model_metal-inspection.tflite"
label_path = "labels.txt"
tflite_model = model_path

# Preprocessing of the data.
input_dir = "to_classify"
images = []
for index, file in enumerate(pathlib.Path(input_dir).iterdir()):
    img = Image.open(file).convert('RGB')
    images.append(img)

# Load the TFLite model and allocate tensors.
interpreter = Interpreter(tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
tensor_index = input_details[0]['index']
input_data = np.asarray(images)
input_data32 = input_data.astype('float32')/255
interpreter.set_tensor(tensor_index, input_data32)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
output_data = interpreter.get_tensor(output_details[0]['index'])
label_id = np.argmax(output_data[0])
prob = np.max(output_data)

# Label the output.
with open(label_path, 'r') as file:
    labels = [line.strip() for i, line in enumerate(file.readlines())]
classification_label = labels[label_id]
print('Class:\t', classification_label, '\t\tProbability: \t', round(prob*100, 2), ' %')
