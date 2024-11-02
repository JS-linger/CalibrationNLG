import os
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

def enumerate_prefixes(text, prefix_length=None):
    """
    Enumerates all prefixes for a given text at the token (word) level.
    Optionally, limit to a specific prefix length (in tokens).
    """
    tokens = text.split()

    if prefix_length is None:
        prefix_length = len(tokens)

    prefixes = [" ".join(tokens[:i]) for i in range(1, min(len(tokens) + 1, prefix_length + 1))]
    return prefixes

def weighted_exponential_sampler(prefixes, sample_size=3, seed=42, decay_rate=0.6):
    """
    Samples `sample_size - 2` unique prefix positions from the input list of prefixes using an exponential weighting scheme.
    Always includes the first and last prefixes (anchors).
    """
    seq_length = len(prefixes)
    
    if seq_length <= sample_size:
        return prefixes
    
    if seed is not None:
        random.seed(seed)

    anchors = [0, seq_length - 1]

    weights = np.exp(-decay_rate * np.arange(1, seq_length - 1))
    weights /= np.sum(weights)

    sampled_positions = np.random.choice(np.arange(1, seq_length - 1), size=sample_size - 2, replace=False, p=weights)
    all_positions = sorted(anchors + list(sampled_positions))

    return [prefixes[i] for i in all_positions]

def process_dataset(df, max_prefix_length=None, sample_size=10, seed=42, decay_rate=0.6, subsample=True):
    """
    Processes a dataset to generate prefix enumerations and, optionally, apply random sampling for training set.
    """
    df['prefixes'] = df.iloc[:, 0].apply(lambda text: enumerate_prefixes(text.strip(), max_prefix_length))

    if subsample:
        df['prefixes'] = df['prefixes'].apply(lambda prefixes: weighted_exponential_sampler(prefixes, sample_size, seed, decay_rate))

    df_exploded = df[['index', 'prefixes', 'labels']].explode('prefixes')

    return df_exploded

def map_labels_to_integers(df, label_column):
    unique_labels = df[label_column].unique()
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    df['labels'] = df[label_column].map(label_map).astype(int)
    return df, label_map

def load_preprocessed_data(df, label_column= 'genre', test_size=0.2, seed=42, max_prefix_length=None, sample_size=10, decay_rate=0.6):
    """
    Load the preprocessed data and split it into train and validation sets.
    """
    df['index'] = df.index

    # Map labels to integers
    df, label_map = map_labels_to_integers(df, label_column)

    df_train, df_validation = train_test_split(df, test_size=test_size, random_state=seed)

    # Process training set with subsampling
    df_train_processed = process_dataset(df_train, max_prefix_length=max_prefix_length,
                                         sample_size=sample_size, seed=seed, decay_rate=decay_rate, subsample=True)
    
    # Process validation set without subsampling
    df_validation_processed = process_dataset(df_validation, max_prefix_length=max_prefix_length,
                                              sample_size=sample_size, seed=seed, decay_rate=decay_rate, subsample=False)

    dataset_train = Dataset.from_pandas(df_train_processed)
    dataset_validation = Dataset.from_pandas(df_validation_processed)

    data = DatasetDict({
        'train': dataset_train,
        'validation': dataset_validation
    })

    return data, label_map