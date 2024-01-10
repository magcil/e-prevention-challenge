# e-prevention-challenge

## 1. Intro
This is [MagCIL's](http://magcil.github.io) approach for the 1st track of the "2nd e-Prevention challenge: Psychotic and Non-Psychotic Relapse Detection using Wearable-Based Digital Phenotyping". Our method ranked second with $\text{ROC}_{\text{AUC}}=0.651$ and $\text{PR}_{\text{AUC}}=0.642$ on the final test dataset of the challenge. 

## 2. About the method

## 3. Code installation and usage

### 3.1 Environment setup

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

### 3.2 Feature extraction

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

### 3.3 Training pipeline

