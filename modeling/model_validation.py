import time
import numpy as np

from modeling.models.model_configs import *
from modeling.utils import root_mean_squared_error, read_parquet


class OneSetValidation:
    def __init__(self, target_col, model, metric=root_mean_squared_error):
        self.target_col = target_col
        self.model = model
        self.metric = metric

    def run(self, train, test, copy=False):
        y_train, train = self.data_target_split(train, copy)
        y_test, test = self.data_target_split(test, copy)

        self.model.fit(train, y_train, (test, y_test))

        predictions = self.model.transform(test)

        result = self.metric(y_test, predictions.clip(0, 20))

        return result

    def data_target_split(self, dataset, copy):
        y = dataset[[self.target_col]]
        if copy:
            dataset = dataset.drop(self.target_col, axis=1)
        else:
            dataset.drop(self.target_col, axis=1, inplace=True)

        return y, dataset


class TimeBasedValidation:
    def __init__(self, target_col, model, time_column_name, metric=root_mean_squared_error):
        self.time_column_name = time_column_name
        self.one_set_validation = OneSetValidation(target_col, model, metric)

    def run(self, train_set, train_period, test_period, num_splits, copy=False):
        time_values = train_set[self.time_column_name].drop_duplicates().sort_values(ascending=False)

        scores = []

        for i in range(num_splits):
            test_max, test_min = time_values.iloc[i], time_values.iloc[i+test_period-1]
            train_max, train_min = time_values.iloc[i+test_period], time_values.iloc[i+test_period+train_period-1]

            train_subset = train_set[train_set[self.time_column_name].between(train_min, train_max)]
            test_subset = train_set[train_set[self.time_column_name].between(test_min, test_max)]

            scores.append(self.one_set_validation.run(train_subset, test_subset, copy))

        result = np.array(scores)

        return {"partial_results": result, "mean": result.mean(), "std": result.std()}


if __name__ == "__main__":
    validation_train_set_path = "data/processed_validation/train.parquet"
    validation_test_set_path = "data/processed_validation/test.parquet"
    target_col = "item_cnt_month"

    valid_train_set = read_parquet(validation_train_set_path)
    valid_test_set = read_parquet(validation_test_set_path)

    print("Read data")

    # Choose the model! Look at model_configs.py for possible options
    # model = get_model(simple_xgb)
    # model = get_model(pytorch_embed)
    model = get_model(lstm_net)

    start = time.time()

    one_set_validation = OneSetValidation(
        target_col,
        model
    )

    single_eval_result = one_set_validation.run(valid_train_set, valid_test_set)
    print("Single evaluation result:\t{}".format(single_eval_result))

    end = time.time()

    print("Time taken: {}s".format(str(end - start)))

    # For now, as the RNN model is memory extensive, give up on the multi period validation.
    # full_train_set = valid_train_set.append(valid_test_set, ignore_index=True)
    # multi_period_validation = TimeBasedValidation(target_col, model, 'date_block_num')
    # multi_eval_result = multi_period_validation.run(full_train_set, 29, 1, 5, True)
    # print("Multi evaluation result:\n", multi_eval_result)
