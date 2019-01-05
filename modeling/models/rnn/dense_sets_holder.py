import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DenseSetsHolder:
    def __init__(self, items_path, items_categories_path, shops_path,
                 items_train_ids, items_categories_train_ids, shops_train_ids):
        self.items = DenseSetsHolder.preproc_dataset(
            pd.read_parquet(items_path), items_train_ids
        )
        self.items_categories = DenseSetsHolder.preproc_dataset(
            pd.read_parquet(items_categories_path), items_categories_train_ids
        )
        self.shops = DenseSetsHolder.preproc_dataset(
            pd.read_parquet(shops_path), shops_train_ids
        )

    @staticmethod
    def standardize(dataset):
        scaler = StandardScaler()
        scaler.fit(dataset)
        return scaler

    @staticmethod
    def preproc_dataset(dataset, train_ids):
        id_attr_name, ids = train_ids
        fit_dataset = dataset.merge(ids.reset_index(), on=id_attr_name).\
            sort_values(id_attr_name).drop([id_attr_name, "index"], axis=1)
        dataset = dataset.sort_values(id_attr_name).drop(id_attr_name, axis=1)
        scaler = DenseSetsHolder.standardize(fit_dataset.values)
        dataset = torch.Tensor(scaler.transform(dataset.values))

        return dataset
