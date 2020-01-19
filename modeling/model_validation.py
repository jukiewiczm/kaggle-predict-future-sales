import time
import numpy as np
from enum import Enum
from modeling.models.model_configs import *
from modeling.utils import root_mean_squared_error, read_parquet


class StandardValidation:
    def __init__(self, target_col, model, metric=root_mean_squared_error):
        self.target_col = target_col
        self.model = model
        self.metric = metric

    def run(self, train, test, inner_train_validation=True, copy=False):
        """
        :param inner_train_validation if this flag is set, you will see validation result during training, but it will
        take longer to train the model
        :param copy: if input sets should stay untouched (not modified in place) after the process, use this flag.
        Warning: the process will take more memory with this flag on.
        """
        y_train, train = self.data_target_split(train, copy)
        y_test, test = self.data_target_split(test, copy)

        if inner_train_validation:
            self.model.fit(train, y_train, (test, y_test))
        else:
            self.model.fit(train, y_train)

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
        self.one_set_validation = StandardValidation(target_col, model, metric)

    def run(self, train_set, train_period, test_period, num_splits, inner_train_validation=True):
        time_values = train_set[self.time_column_name].drop_duplicates().sort_values(ascending=False)

        scores = []
        for i in range(num_splits):
            test_max, test_min = time_values.iloc[i], time_values.iloc[i+test_period-1]
            train_max, train_min = time_values.iloc[i+test_period], time_values.iloc[i+test_period+train_period-1]

            # Explicit copy to avoid the "assignment on copy" warning
            train_subset = train_set[train_set[self.time_column_name].between(train_min, train_max)].copy()
            test_subset = train_set[train_set[self.time_column_name].between(test_min, test_max)].copy()

            scores.append(self.one_set_validation.run(train_subset, test_subset, inner_train_validation))
            del train_subset, test_subset

        result = np.array(scores)

        return {"partial_results": result, "mean": result.mean(), "std": result.std()}


class ValidationType(Enum):
    STANDARD = 1
    TIME_BASED = 2
    BOTH = 3  # Might be memory expensive


if __name__ == "__main__":
    validation_train_set_path = "data/processed_validation/train.parquet"
    validation_test_set_path = "data/processed_validation/test.parquet"
    target_col = "item_cnt_month"

    valid_train_set = read_parquet(validation_train_set_path)
    valid_test_set = read_parquet(validation_test_set_path)

    # Choose validation type
    validation_type = ValidationType.STANDARD

    print("Read data")

    # Choose the model! Look at model_configs.py for possible options
    # model = get_model(simple_xgb)
    # model = get_model(pytorch_embed)
    model = get_model(lstm_net)

    start = time.time()

    if validation_type == ValidationType.STANDARD or validation_type == ValidationType.BOTH:
        standard_validation = StandardValidation(
            target_col,
            model
        )

        # If both, here we need a copy
        single_eval_result = standard_validation.run(
            valid_train_set,
            valid_test_set,
            copy=validation_type == ValidationType.BOTH
        )
        print("Single evaluation result:\t{}".format(single_eval_result))

    if validation_type == ValidationType.TIME_BASED or validation_type == ValidationType.BOTH:
        full_train_set = valid_train_set.append(valid_test_set, ignore_index=True)
        del valid_train_set, valid_test_set

        multi_period_validation = TimeBasedValidation(target_col, model, 'date_block_num')
        multi_eval_result = multi_period_validation.run(full_train_set, 30, 1, 4, False)
        print("Multi evaluation result:\n", multi_eval_result)

    end = time.time()
    print("Time taken: {}s".format(str(end - start)))
