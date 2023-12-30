{
    "feature_mapping": {
    "hrm": [
        "heartRate_nanmean", "rRInterval_nanmean", "rRInterval_rmssd", "rRInterval_sdnn"
    ],
    "gyr": ["gyr_mean", "gyr_std"],
    "linacc": ["acc_mean", "acc_std"],
    "time": ["cos_t", "sin_t"]
    },
    "track_id": 1,
    "patients":[1, 2, 3, 4, 5, 6, 7, 8, 9],
    "epochs": 100,
    "patience": 10,
    "models": ["AnomalyTransformer"],
    "learning_rate": 1e-4,
    "upsampling_size": 120,
    "prediction_upsampling": 120,
    "window_size": 32,
    "file_format": ".parquet",
    "split_ratio": 0.8,
    "batch_size": 128,
    "pretrained_models": "pretrained_models/",
    "one_class_test": 1,
    "num_workers": 8

}