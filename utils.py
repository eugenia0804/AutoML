import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

TRAIN_SIZE = 5000
TEST_SIZE = 1000

def load_data():
    train_data = np.load("data/train_data.npy", allow_pickle=True)
    test_data = np.load("data/test_data.npy", allow_pickle=True)

    # combine all training shards
    X_list = [np.array(c["images"]) for c in train_data]
    y_list = [np.array(c["labels"]) for c in train_data]

    X_full = np.concatenate(X_list, axis=0).astype(np.float32)
    y_full = np.concatenate(y_list, axis=0).astype(np.int64)

    # keep only labels 0–9
    keep = (y_full >= 0) & (y_full <= 9)
    X_full = X_full[keep]
    y_full = y_full[keep]

    # shuffle before slicing
    perm = np.random.permutation(len(y_full))
    X_full = X_full[perm]
    y_full = y_full[perm]

    # take TRAIN + VAL
    X_train = X_full[:TRAIN_SIZE]
    y_train = y_full[:TRAIN_SIZE]

    # test set
    X_test = np.array(test_data[0]["images"]).astype(np.float32)
    y_test = np.array(test_data[0]["labels"]).astype(np.int64)

    keep_test = (y_test >= 0) & (y_test <= 9)
    X_test = X_test[keep_test][:TEST_SIZE]
    y_test = y_test[keep_test][:TEST_SIZE]

    # add channel dimension
    X_train = X_train[:, None, :, :]
    X_test = X_test[:, None, :, :]

    print("Train size:", X_train.shape, "Labels range:", y_train.min(), "-", y_train.max())
    print("Test size:", X_test.shape, "Labels range:", y_test.min(), "-", y_test.max())

    return X_train, y_train, X_test, y_test

def make_loaders(train_x, train_y, valid_x, valid_y, train_batch_size, valid_batch_size=1024):
    train_ds = TensorDataset(torch.tensor(train_x, dtype=torch.float32),
                             torch.tensor(train_y, dtype=torch.long))
    valid_ds = TensorDataset(torch.tensor(valid_x, dtype=torch.float32),
                             torch.tensor(valid_y, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=valid_batch_size, shuffle=False)
    return train_loader, valid_loader

def make_test_loader(x, y, batch_size=1024):
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
