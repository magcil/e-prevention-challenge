import pandas as pd
import os

def normalize_cols(base_path, features_paths, out_dir):
    TIME_RELATED = ['DateTime', 'sin_t', 'cos_t']
    for f in features_paths:
        print('features paths:', f)
        day_df = pd.read_parquet(os.path.join(base_path, f), engine='fastparquet')

        for col in day_df.columns:
            if col != "DateTime":
                for i in day_df[col].index:
                    day_df[col][i] = (day_df[col][i] - day_df[col].min()) / (day_df[col].max() - day_df[col].min())

        filename = f'norm_{f}'
        out_dir = os.path.dirname(base_path) + '/norm_features'

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)


        out_file = os.path.join(out_dir, filename)
        day_df.to_parquet(out_file, engine='fastparquet')
        norm_base_path = out_dir
        normalized_feat_paths = os.listdir(out_dir)