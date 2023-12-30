{
    "feature_mapping": {
    "hrm": [
        "heartRate_nanmean", "rRInterval_nanmean", "rRInterval_rmssd", "rRInterval_sdnn"
    ],
    "gyr": ["gyr_mean", "gyr_std", "gyr_delta_mean", "gyr_delta_std"],
    "linacc": ["acc_mean", "acc_std", "acc_delta_std"],
    "sleep": ["interval_sleep", "aggr_sleep", "n_sleep"],
    "time": ["cos_t", "sin_t"]
    },
    "track_id": 1,
    "patients":[1, 2, 3, 4, 5, 6, 7, 8, 9],
    "epochs": 100,
    "patience": 10,
    "model": "AnomalyTransformer", #choose between AnomalyTransformer, Autoencoder, AnomalyTransformer_2, Autoencoder_2
    "learning_rate": 1e-3,
    "upsampling_size": 120,
    "prediction_upsampling": 120,
    "window_size": 32,
    "file_format": ".parquet",
    "split_ratio": 0.8,
    "batch_size": 128,
    "pretrained_models": "pretrained_models/",  #where to save the encoder model after training it
    "one_class_test": [1, 1, 1, 1, 1, 1, 1, 1, 1], #use or not svm after encoder per patient
    "num_workers": 4,
    "saved_checkpoint": ["p1_best_modesl.pth",
                         "p2_best_modesl.pth",
                         "p3_best_modesl.pth",
                         "p4_best_modesl.pth",
                         "p5_best_modesl.pth",
                         "p6_best_modesl.pth",
                         "p7_best_modesl.pth",
                         "p8_best_modesl.pth",
                         "p9_best_modesl.pth"],  #saved tranformer or cae best model. In case of training from scratch set this to None (per patient)
    "num_layers": [3, 3, 3, 3, 3, 3, 3, 3, 3], #used only if model is defined as Autoencoder_2 and AnomalyTransformer_2 that are tuned models with custom number of layers (per patient)
    "test_metric": "median" #used only if one_class_test is set to 1, choose between mean, median, percentile, weighted_hard_decision
}
