import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from fastparquet import ParquetFile
from glob import glob


def read_parquet(path):
    file_list = glob("{}/*.parquet".format(path))
    pf = ParquetFile(file_list)
    # Change types for better memory consumption.
    # We also never want the index column as well.
    return pf.to_pandas().astype("float32", copy=False).drop('index', axis=1, errors='ignore')


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def prepare_aggregate_submission(dir_path, out_path, func=lambda x: x.mean(axis=0)):
    """
    Given a directory with various submissions, creates simple ensemble submission by aggregating all the predictions
    (takes an average of them by default).
    """
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".csv.gz")]
    id_col = pd.read_csv(files[0])['ID'].to_numpy()
    predictions = func(np.array([pd.read_csv(f)['item_cnt_month'].to_numpy() for f in files]))
    result = pd.DataFrame({'ID': id_col, 'item_cnt_month': predictions})
    result.to_csv(out_path, index=False, header=True, compression="gzip")
