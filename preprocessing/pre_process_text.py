import os
import random
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

def enumerate_prefixes(text: str, prefix_length: int | None = None) -> list[str]:
    """
    Enumerates all prefixes for a given text at the token (word) level.
    Optionally, limit to a specific prefix length (in tokens).
    """
    tokens = text.split()

    if prefix_length is None:
        prefix_length = len(tokens)

    prefixes = [" ".join(tokens[:i]) for i in range(1, min(len(tokens) + 1, prefix_length + 1))]
    return prefixes

def weighted_exponential_sampler(
    prefixes: list[str], 
    sample_size: int = 3, 
    seed: int | None = 42, 
    decay_rate: float = 0.6
) -> list[str]:
    """
    Samples `sample_size - 2` unique prefix positions from the input list of prefixes 
    using an exponential weighting scheme. Always includes the first and last prefixes.
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

def process_dataset(
    df: pd.DataFrame, 
    max_prefix_length: int | None = None, 
    sample_size: int = 10, 
    seed: int = 42, 
    decay_rate: float = 0.6, 
    subsample: bool = True
) -> pd.DataFrame:
    """
    Processes a dataset to generate prefix enumerations and apply random sampling for training set.
    """

    df['text'] = df.iloc[:, 0].apply(lambda text: enumerate_prefixes(text.strip(), max_prefix_length))

    if subsample:
        df['text'] = df['text'].apply(
            lambda prefixes: weighted_exponential_sampler(prefixes, sample_size, seed, decay_rate)
        )

    df_exploded = df[['index', 'text', 'labels']].explode('text')

    return df_exploded

def map_labels_to_integers(
    df: pd.DataFrame, 
    label_column: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Maps string labels to integers.
    """
    unique_labels = df[label_column].unique()
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    df['labels'] = df[label_column].map(label_map).astype(int)
    return df, label_map

def load_preprocessed_data(
    df: pd.DataFrame, 
    label_column: str = 'genre', 
    test_size: float = 0.2, 
    seed: int = 42, 
    max_prefix_length: int | None = None, 
    sample_size: int = 10, 
    decay_rate: float = 0.6,
    process_prefixes: bool = True
) -> tuple[DatasetDict, dict[str, int]]:
    """
    Load and preprocess data, splitting into train and validation sets.
    
    Args:
        df: Input DataFrame
        label_column: Name of the label column
        test_size: Fraction of data to use for validation
        seed: Random seed
        max_prefix_length: Maximum prefix length for enumeration
        sample_size: Number of prefixes to sample
        decay_rate: Decay rate for exponential sampling
        process_prefixes: Whether to process text into prefixes
    
    Returns:
        Tuple of (DatasetDict containing train and validation sets, label mapping dictionary)
    """
    df['index'] = df.index

    df = df.rename(columns={df.columns[0]: 'text'})

    # Map labels to integers
    df, label_map = map_labels_to_integers(df, label_column)

    # Split into train and validation
    df_train, df_validation = train_test_split(df, test_size=test_size, random_state=seed)

    if process_prefixes:
        # Process training set with subsampling
        df_train_processed = process_dataset(
            df_train, 
            max_prefix_length=max_prefix_length,
            sample_size=sample_size, 
            seed=seed, 
            decay_rate=decay_rate, 
            subsample=True
        )
        
        # Process validation set without subsampling
        df_validation_processed = process_dataset(
            df_validation, 
            max_prefix_length=max_prefix_length,
            sample_size=sample_size, 
            seed=seed, 
            decay_rate=decay_rate, 
            subsample=False
        )
    else:
        # Use original data without prefix processing
        df_train_processed = df_train
        df_validation_processed = df_validation

    dataset_train = Dataset.from_pandas(df_train_processed)
    dataset_validation = Dataset.from_pandas(df_validation_processed)

    data = DatasetDict({
        'train': dataset_train,
        'validation': dataset_validation
    })

    return data, label_map