import tflite_runtime.interpreter as Interpreter
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from sklearn.datasets import load_files

def convert_image_to_array(files):
    images_as_array=[]
    for file in files:
        # Convert to Numpy Array
        images_as_array.append(img_to_array(load_img(file)))
    return images_as_array


def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])


# Input Data Configuration
data_dir = "/home/pi/io-model-test/"
model_path = data_dir + "saved-model_metal-inspection.tflite"
label_path = data_dir + "labels.txt"
tflite_model = model_path

# Preprocessing of the data to ensure the (n, 200, 200, x) array size
input_dir = data_dir + "/to_classify"
x_input = load_dataset(input_dir)
#image = Image.open(input_dir + "/image.bmp").convert('RGB').resize((width, height)) #makes sure to resize to 200x200 if tflite shape comes before
x_test = np.array(convert_image_to_array(x_input))
print('Shape : ',x_test.shape)
x_test = x_test.astype('float32')/255

# Load the TFLite model and allocate tensors.
interpreter = Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
#_, height, width, _ = interpreter.get_input_details()[0]['shape'] #for defining height & size to resize up-front
#tensor_index = interpreter.get_input_details()[0]['index']
#input_tensor = interpreter.tensor(tensor_index)()[0]
#input_tensor[:, :] = image
#Original Input Sample
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
input_data = x_test
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
#output = np.squeeze(interpreter.get_tensor(output_details['index']))
#scale, zero_point = output_details['quantization_parameters']
#output = scale * (output - zero_point)
#ordered = np.argpartition(-output, 1)
#label_id, prob = [(i, output[i]) for i in ordered[:1]]
print(output_data)

# Label the output
with open(label_path, 'r') as file:
    labels = [line.strip() for i, line in enumerate(file.readlines())]
#classification_label = labels[label_id]
