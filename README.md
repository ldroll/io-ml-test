# io-ml-test
This project is for testing the implementation of pre-trained machine learning models using Tensorflow on edge devices like the Raspberry Pi. The trained model is converted to the leightweight tflite format considering performance capabilities.
Since this project shall exist as a comfortable demonstration of the use of edge devices with ML models, the installation and setup process as reduced as much as possible.

## Pi Setup
Pi Setup can be done via the typical setup process of installing OS and connecting to the network. For further instructions, see [RaspPi Documentation](https://www.raspberrypi.com/documentation/computers/getting-started.html).

Make sure to be up to date:
```
sudo apt-get update && audo apt-get upgrade
```

## Download Model and Dependencies
The `.tflite` model as well as the `tflite-runtime` and other dependencies are downloaded by git cloning this repository and installed by executing `setup.sh`.
```
git clone https://github.com/ldroll/io-ml-test
cd io-ml-test
sudo chmod +x setup.sh
sh setup.sh
```

## Trigger the python script per command
The classification function can be called by executing the `classify.py` script which classifies the image contained in the `to_classify` folder.
```
python3 classify.py
```
