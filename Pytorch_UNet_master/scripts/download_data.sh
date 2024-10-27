#!/bin/bash

# Check if Kaggle API credentials are available, and prompt for them if not
if [[ ! -f ~/.kaggle/kaggle.json ]]; then
  read -p "Kaggle username: " USERNAME
  read -s -p "Kaggle API key: " APIKEY
  echo

  mkdir -p ~/.kaggle
  echo "{\"username\":\"$USERNAME\",\"key\":\"$APIKEY\"}" > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
fi

# Upgrade kaggle package
pip install --upgrade kaggle

# Download and organize the images
kaggle competitions download -c carvana-image-masking-challenge -f train_hq.zip && unzip -q train_hq.zip -d data/imgs/ && rm train_hq.zip

# Download and organize the masks
kaggle competitions download -c carvana-image-masking-challenge -f train_masks.zip && unzip -q train_masks.zip -d data/masks/ && rm train_masks.zip
