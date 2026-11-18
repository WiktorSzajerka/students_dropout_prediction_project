import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
from ARM_Net.utils.uci_utils import nsample_bsz_mapping


def csv_loader(csv_path, batch_size=0, valid_perc=0.15, workers=1,
               verbose=True, header='infer', sep=',', seed=0):
    '''
    :param csv_path:        Path to the CSV file (features in all columns but the last, target in the last)
    :param batch_size:      Batch size (0 -> chosen automatically from the number of samples)
    :param test_perc:       fraction of the whole dataset used as the test set
    :param valid_perc:      valid percentage split from train (default 0, whole train set)
    :param workers:         the number of subprocesses to load data
    :param header:          'infer' if the CSV has a header row, None if it does not
    :param sep:             CSV separator
    :param seed:            random seed for the splits
    :return:                train/valid/test loader (train_loader has nclass, nfeat, nsample)
    '''

    def split(X, y, split_perc):
        try:
            return sklearn.model_selection.train_test_split(
                X, y, test_size=split_perc, random_state=seed, stratify=y)
        except ValueError:
            return sklearn.model_selection.train_test_split(
                X, y, test_size=split_perc, random_state=seed)

    def make_loader(X, y, transformer=None, batch_size=64, drop_last=False):
        if transformer is None:
            transformer = sklearn.preprocessing.StandardScaler()
            transformer.fit(X)
        X = transformer.transform(X).astype(np.float32)
        return DataLoader(
            dataset=TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=drop_last
        ), transformer

    # Load the whole file
    df = pd.read_csv(csv_path, header=header, sep=sep)
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_raw = df.iloc[:, -1].to_numpy()

    # Encode targets to 0..nclass-1
    label_encoder = sklearn.preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y_raw).astype(np.int64)
    nclass = len(label_encoder.classes_)


    # Train / valid split
    if valid_perc > 0:
        Xtrain, Xvalid, ytrain, yvalid = split(X, y, valid_perc)
        if batch_size == 0:
            batch_size = nsample_bsz_mapping(Xtrain.shape[0])
        
        train_loader, transformer = make_loader(Xtrain, ytrain, batch_size=batch_size, drop_last=False)
        valid_loader, _ = make_loader(Xvalid, yvalid, transformer, batch_size=batch_size, drop_last=False)

        if verbose:
                print(f'{csv_path}: {len(ytrain)} samples loaded (bsz {batch_size}).')

    else:
        train_loader, transformer = make_loader(X, y, batch_size=1, drop_last=False)
        train_loader.nclass, train_loader.nfeat, train_loader.nsample = nclass, X.shape[-1], X.shape[0]
        train_loader.transformer = transformer          # StandardScaler fitted on the train set
        train_loader.label_encoder = label_encoder 
        if verbose:
                print(f'{csv_path}: {len(y)} test samples loaded (bsz {batch_size}).')
        return train_loader

        


    


    train_loader.nclass, train_loader.nfeat, train_loader.nsample = nclass, Xtrain.shape[-1], Xtrain.shape[0]
    train_loader.transformer = transformer          # StandardScaler fitted on the train set
    train_loader.label_encoder = label_encoder      # maps original labels <-> 0..nclass-1

    return train_loader, valid_loader


if __name__ == "__main__":
    train_loader, valid_loader, test_loader = csv_loader(csv_path=r"data\transformed_data.csv")

