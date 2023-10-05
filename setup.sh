#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="./"
else
  DATA_DIR="$1"
fi

# Install Python dependencies.
python3 -m pip install pip --upgrade
python3 -m pip install -r requirements.txt

# Download .tflite model (private folder because size > 25 GB).
FILE=${DATA_DIR}/saved-model_metal-inspection.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L 'https://drive.google.com/uc?export=download&id=1uFwNPj7hZ9tE00fUzr5Fiup_U4_2xRe2' \
    -o ${FILE}
fi
echo -e "Downloaded files are in ${DATA_DIR}"
