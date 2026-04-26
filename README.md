# Environmental Sound Classification (Statistical Features + ML)

## Setup in VSCode

1. Open VSCode
2. Extract this folder
3. Open the folder in VSCode
4. Open terminal and run:

pip install -r requirements.txt

## Dataset

Download ESC-50 and place it like this:

env_sound_cls/
  data/
    ESC-50/
      audio/
      meta/esc50.csv

## Run

Extract features:

python src/extract_features.py --esc_root data/ESC-50

Train models:

python src/train_models.py

Results will be saved in the results/ folder.