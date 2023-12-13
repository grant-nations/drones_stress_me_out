import os
import json
import numpy as np
import pandas as pd
from utils.io import generate_unique_filename


def normalize(df):
    X = df.values
    mean = np.mean(X, axis=0)
    X -= mean  # zero-center the data
    std = np.std(X, axis=0)
    X /= std  # normalize the data
    return pd.DataFrame(X), mean, std


def process_data():
    # find suffix of most recent data
    suffix = ""
    while os.path.exists(os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 'processed', f"demo_data{suffix}.csv")):
        suffix = f"-{int(suffix[1:]) + 1}" if suffix else "-1"

    demo_data_filename = f"demo_data{suffix}.csv"
    input_data_filename = f"drone_and_bio_input{suffix}.csv"
    labels_data_filename = f"stress_labels{suffix}.csv"

    demo_features_filename = 'demo_features.json'

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    raw_data_dir = os.path.join(data_dir, 'raw')
    processed_data_dir = os.path.join(data_dir, 'processed')

    # LOAD DEMOGRAPHIC DATA

    demo_df = pd.read_csv(os.path.join(raw_data_dir, demo_data_filename))

    # ENCODE DEMOGRAPHIC DATA

    features = None
    with open(os.path.join(data_dir, demo_features_filename), 'r') as f:
        features = json.load(f)

    for feature_name, feature_values in features.items():
        if isinstance(feature_values, list):
            encoding_vec = list(range(len(feature_values)))
            demo_df[feature_name].replace(feature_values, encoding_vec, inplace=True)

    demo_df = demo_df.astype('float64')

    print("Demo encoding sanity check:\n")
    print(demo_df.head())

    # NORMALIZE DEMOGRAPHIC DATA

    max_val = np.max(demo_df, axis=1).item()
    min_val = np.min(demo_df, axis=1).item()
    demo_df = (demo_df - min_val) / (max_val - min_val)

    print("Demo normalization sanity check:\n")
    print(demo_df.head())

    # SAVE DEMOGRAPHIC DATA

    filename = os.path.join(processed_data_dir, demo_data_filename)
    filename = generate_unique_filename(filename)
    demo_df.to_csv(filename, index=False)

    # LOAD INPUT DATA

    input_df = pd.read_csv(os.path.join(raw_data_dir, input_data_filename))

    # NORMALIZE INPUT DATA

    input_df, mean, std = normalize(input_df)

    print("Input normalization sanity check:\n")
    print(input_df.head())

    # SAVE INPUT DATA

    filename = os.path.join(processed_data_dir, input_data_filename)
    filename = generate_unique_filename(filename)
    input_df.to_csv(filename, index=False)

    # SAVE MEAN AND STD

    mean_std_dict = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }

    filename = os.path.join(processed_data_dir, 'mean_std.json')
    filename = generate_unique_filename(filename)
    with open(filename, 'w') as f:
        json.dump(mean_std_dict, f)

    # LOAD STRESS LABELS

    output_df = pd.read_csv(os.path.join(raw_data_dir, labels_data_filename))

    # ROUND OUTPUT DATA TO NEAREST INTEGER

    output_df = output_df.round()

    print("Output rounding sanity check:\n")
    print(output_df.head())

    # SAVE OUTPUT DATA

    filename = os.path.join(processed_data_dir, labels_data_filename)
    filename = generate_unique_filename(filename)
    output_df.to_csv(filename, index=False)


if __name__ == "__main__":
    process_data()
