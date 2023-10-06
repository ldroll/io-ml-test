# io-ml-test
This project is for testing the implementation of pre-trained machine learning models using Tensorflow on edge devices like the Raspberry Pi. The trained model is converted to the leightweight tflite format considering performance capabilities.
Since this project shall exist as a comfortable demonstration of the use of edge devices with ML models, the installation and setup process as reduced as much as possible.

The machine learning model is based on the ["Metal Surface Defects" dataset](https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data/code) from Kaggle.

## Pi Setup
Pi Setup can be done via the typical setup process of installing OS and connecting to the network. For further instructions, see [RaspPi Documentation](https://www.raspberrypi.com/documentation/computers/getting-started.html).

Make sure to be up to date:
```
sudo apt-get update && audo apt-get upgrade
```

## Download Model and Dependencies
By git cloning this repository and executing `setup.sh`, the `.tflite` model will be downloaded. Due to file size restrictions, the model is not included in this repo but outsourced to a Google drive folder. `tflite-runtime` and other dependencies will also automatically installed.
```
git clone https://github.com/ldroll/io-ml-test
cd io-ml-test
sudo chmod +x setup.sh
sudo /.setup.sh
```
In case this repo is private, you need to include your PAT for git cloning. For instructions, see [Stackoverflow - PAT](https://stackoverflow.com/questions/2505096/clone-a-private-repository-github).

## Trigger the python script per command
The classification function can be called by executing the `classify.py` script which classifies the image contained in the `to_classify` folder.
```
python3 classify.py
```
