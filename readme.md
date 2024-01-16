# e-prevention-challenge

## Authors
Panagiotis Kaliosis
Sofia Eleftheriou
Christos Nikou
Thodoris Giannakopoulos

## 1. About
This code corresponds to [MagCIL's](http://magcil.github.io) approach for the 1st track of the "2nd e-Prevention challenge: Psychotic and Non-Psychotic Relapse Detection using Wearable-Based Digital Phenotyping". First we present our approach for preprocessing and extracting features from the wearable's raw data. We then propose a Transformer model for learning self-supervised representations from augmented features,  trained on data from non-relapse days from each of the 9 patients of the challenge. We adopt two unsupervised  methods for detecting relapse days as outliers. A separate unsupervised model is tuned for each patient using the validation data of the challenge. Our method ranked 2nd with ROC_AUC=0.651 and PR_AUC=0.642 on the final test dataset of the challenge. 

## 2. Code installation and usage

### 2.1 Environment setup

To run the experiments and reproduce the results first create a conda environment with python 3.9 by typing

```bash
conda create -n e-prevention python=3.9
```

Activate the environment by typing `conda activate e-prevention` and install the requirements with `pip install -r requirements.txt`

The place the data of the challenge inside the "data" folder of this repository with the following structure:

```
├── data
│   ├── track_01
│   │   ├── P1
│   │   │   ├── test_0 
│   │   │   │   ├── gyr.parquet
│   │   │   │   ├── hrm.parquet
│   │   │   │   ├── linacc.parquet
│   │   │   │   ├── sleep.parquet
│   │   │   │   ├── step.parquet
│   │   │   ├── test_1
│   │   │   │   ├── ...
│   │   │   ├── test_2
│   │   │   ├── ...
│   │   │   ├── val_1
│   │   ├── ...
│   │   ├── P9
│   │   │   ├── ...
│   ├── track_02
│   │   ├── ...
```   

### 2.2 Feature extraction

To extract the features from the raw data from track 1 described in the manuscript run the following command:

```bash
python preprocess/extract_all_features.py --track 1 --patients 1 2 3 4 5 6 7 8 9 --dtypes hrm gyr linacc sleep --output_format csv
```

The argument dtypes controls the sensors to be used for feature extraction (heart rate, gyroscope, linear accelerometer, and sleep measurements). There is also an option for steps but we did not use these measurements in the experiments. Similarly, to extract the respective features for track 2 you can type

```bash
python preprocess/extract_all_features.py --track 2 --patients 1 2 3 4 5 6 7 8 --dtypes hrm gyr linacc sleep --output_format csv
```
`What to expect:` The above commands will create a folder "features" in each of patient's split (train, val, test) and place inside this folder the features of each day of the available data in a csv format that allows for further inspection of the features. The csv files will contain 16 features along with their corresponding timestamp for the current 5-Min interval. 

`Missing values`: If for a given day the measurements of a sensor are missing (e.g. heart rate) then the corresponding values will be NaN by default. On training set these NaN values will be spotted and will be set to zero, a mask for these positions will be created to exclude these features from the calculation of the reconstruction loss.

`Faster feature extraction`: The argument `patients` allows you to extract the features for each patient in parallel. For example you can start 2 processes by typing 

```bash
python preprocess/extract_all_features.py --track 1 --patients 1 2 3 4 5 --dtypes hrm gyr linacc sleep --output_format csv
```
and

```bash
python preprocess/extract_all_features.py --track 1 --patients 5 6 7 8 9 --dtypes hrm gyr linacc sleep --output_format csv
```
However, too many seperate processes might blow up your RAM due to the size of the parquet files containing the data.

### 2.3 Training pipeline

To start training the transformer models for each patient in track 1 run the command:

```bash
python src/end_to_end_evaluation.py --config config/track_1_end_to_end_config.json
```
This will output a `.pt` file in `pretrained_models` for each patient containing the respective weights of the trained transformer. Furthermore, a folder `svms` will be created containing the standard scaler and svm for each patient.

### 3.4 Generate submissions

To generate the submissions for track 1 and evaluate the results on the validation data edit the `track_1_submission_config.json` and place the relative path for each patient's `.pt` file from step 3.3. Then run the command

```bash
python src/generate_submissions.py --config config/track_1_submission_config.json
```

## 3. References

Our Transformer component is based on the "SiT: Self-supervised vIsion Transformer" paper [1]. Moreover, this repo is part of our participation in the 2nd e-Prevention Signal Processing Grand Challenge [2].

<a id="1">[1]</a> 
S. Atito, M. Awais, J. Kittler (2021). 
SiT: Self-supervised vIsion Transformer. 
arXiv preprint arXiv:2104.03602.

<a id="2">[2]</a> 
P. P. Filntisis, N. Efthymiou, G. Retsinas, A. Zlatintsi, C. Garoufis, T. Sounapoglou, P. Tsanakas, N. Smyrnis, P. Maragos (2024). 
“The 2nd e-prevention challenge: Psychotic and non-psychotic relapse detection using wearable-based digital phenotyping,”. 
Proceedings of the International Conference on Acoustics, Speech and Signal Processing. IEEE, Seoul, South Korea, 2024.


