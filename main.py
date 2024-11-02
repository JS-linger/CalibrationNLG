import pandas as pd
from preprocessing.pre_process_text import load_preprocessed_data, map_labels_to_integers
from models.fudge.fudge_train import train_fudge_model

import importlib
import preprocessing.pre_process_text
importlib.reload(preprocessing.pre_process_text)

import models.fudge.fudge_train
importlib.reload(models.fudge.fudge_train)

if __name__ == "__main__":
    df = pd.read_csv('data/tagmybook/data.csv')[["synopsis","genre"]]
    LABEL_COLUMN = 'genre'

    df, label_map = map_labels_to_integers(df, LABEL_COLUMN)
    num_labels = len(label_map)

    data, label_map = load_preprocessed_data(df, sample_size=50, decay_rate=0.1, seed=42)

    # Train the model
    train_fudge_model(
        data,
        model_output_dir='outputs/fudge_model',
        label_column=LABEL_COLUMN,
        num_labels=num_labels,
        epochs=3,
        test_size=0.2,
        seed=42
    )





