import pandas as pd
from preprocessing.pre_process_text import load_preprocessed_data, map_labels_to_integers
from models.fudge.train_fudge_autoregressive import train_fudge_model

import importlib
import preprocessing.pre_process_text
importlib.reload(preprocessing.pre_process_text)

import models.fudge.train_fudge_autoregressive
importlib.reload(models.fudge.train_fudge_autoregressive)

if __name__ == "__main__":
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    # Load data
    df = pd.read_csv('data/tagmybook/data.csv')[["synopsis", "genre"]]
    LABEL_COLUMN = 'genre'

    # Load preprocessed data (includes label conversion)
    data, label_map = load_preprocessed_data(
        df, 
        label_column=LABEL_COLUMN,
        decay_rate=0.1, 
        seed=42, 
        process_prefixes=False
    )
    
    num_labels = len(label_map)
    print(f"Number of unique labels: {num_labels}")
    print(f"Label mapping: {label_map}")

    # Train the model
    train_fudge_model(
        dataset=data,
        model_output_dir='outputs/fudge_model',
        label_column='labels',
        label_mapping=label_map,
        num_labels=num_labels,
        epochs=10,
        seed=42,
        model_name='Qwen/Qwen2.5-0.5B'
    )





