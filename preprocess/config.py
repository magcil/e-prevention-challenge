FEATURE_NAMES = {
    "hrm": [
        'heartRate_nanmean', 'rRInterval_nanmean', 'rRInterval_rmssd', 'rRInterval_sdnn',
        'rRInterval_lombscargle_power_high'
    ],
    "gyr": ['gyr_mean', 'gyr_std', 'gyr_delta_mean', 'gyr_delta_std'],
    "linacc": ['acc_mean', 'acc_std', 'acc_delta_mean', 'acc_delta_std'],
    "sleep": ['interval_sleep', 'aggr_sleep', 'n_sleep']
}
