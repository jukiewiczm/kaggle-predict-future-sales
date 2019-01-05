import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, OrderedDict
from modeling.models.rnn.rnn_dataset import RNNDataset


class RNNDataPreprocessor:
    """ This class handles data preparation and preprocessing for lstm model. """

    def __init__(self, dense_sets_holder):

        self.index_colnames = ['item_id', 'item_category_id', 'shop_id']
        self.data_colnames = None

        self.dense_sets_holder = dense_sets_holder
        self.reference_dict = None

        self.preproc_mean = None
        self.preproc_std = None

        self.standardizer = StandardScaler()

    def preprocess_common(self, dataset, fit_standardizer=True):
        # Some columns have so big values comparing to others that I decided to log transform them
        for col in dataset.columns:
            if ("price" in col) or ("previously" in col):
                dataset[col] = np.log(dataset[col] + 1e-3)

        if fit_standardizer:
            self.standardizer.fit(dataset[self.data_colnames].values)

        # Use the standardizer
        dataset[self.data_colnames] = self.standardizer.transform(dataset[self.data_colnames].values)

        # Fill NA's introduced in dataset preparation procedure in Spark
        dataset.fillna(0, inplace=True)

        # Reindex the DataFrame appropriately, in place
        for i, colname in enumerate(self.index_colnames + self.data_colnames):
            col = dataset.pop(colname)
            dataset.insert(i, colname, col)

        # Keeping it in numpy from this point
        return dataset.to_numpy()

    def preprocess_train_dataset(self, dataset, y_dataset):
        # Separate index column names from data col names
        self.data_colnames = sorted(list(set(dataset.columns.tolist()) - set(self.index_colnames)))

        # When preprocessing training, we care about the time order (as we'll be creating time series), hence the sort.
        # In addition, the copy here is made so source dataset will not be changed.
        dataset = dataset.sort_values(['date_block_num'])
        y_dataset = y_dataset.values[dataset.index].squeeze().tolist()

        dataset = self.preprocess_common(dataset)

        # Adding observations to dict to keep the "shop_id, item_id" pairs together
        dataset_dict = defaultdict(list)

        for row in zip(dataset, y_dataset):
            # No more column names, but this represents (shop_id, item_id)
            key = (row[0][2], row[0][0])
            dataset_dict[key].append(row)

        # If processing train set, the dict is kept for reference when filling in the test set
        self.reference_dict = dataset_dict

        # Prepare tensor-based RNN dataset
        inputs, labels, lens = [], [], []

        for row in dataset_dict.values():
            row_series, label_series = tuple(zip(*row))
            inputs.append(torch.Tensor(row_series))
            labels.append(torch.Tensor(label_series))
            lens.append(len(row_series))

        lstm_dataset = RNNDataset(inputs, lens, labels, self.dense_sets_holder)

        return lstm_dataset

    def preprocess_test_dataset(self, dataset):
        assert self.reference_dict is not None, "Train dataset must be preprocessed before test dataset."

        # Don't want to modify the source datasets just in case, have to survive this copy
        dataset = dataset.copy()
        dataset = self.preprocess_common(dataset, False)

        # In test data, the observation is always the last one, so sorting is not required, but we care about rows
        # position as it's expected for predictions to keep the order.
        dataset_dict = OrderedDict()

        # Fill the dict mentioned previously
        for row in dataset:
            # Similar to train procedure, but append to key if it exists, or create new list otherwise.
            key = (row[2], row[0])
            key_data = [data[0] for data in self.reference_dict[key]] if key in self.reference_dict.keys() else list()
            key_data.append(row.tolist())
            dataset_dict[key] = key_data

        # Prepare tensor-based RNN dataset
        inputs = [torch.Tensor(line) for line in dataset_dict.values()]
        lens = [x.shape[0] for x in inputs]
        lstm_dataset = RNNDataset(inputs, lens, None, self.dense_sets_holder)

        return lstm_dataset
