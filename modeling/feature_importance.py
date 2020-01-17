import pandas as pd
from eli5.permutation_importance import get_score_importances
import numpy as np
from modeling.models.model_configs import *
from modeling.utils import root_mean_squared_error, read_parquet


# Code snippet for measuring feature importance by permutations using eli5.
# Helped me find some bugs in feature design and to design new features as well.
# Since the model architecture is pretty custom and unique, evaluating on part of train data or evaluating the
# embeddings require significant amount of magic. I decided to evaluate embeddings as a whole (not column by column)
# as I believe it makes most sense. You can choose whether embeddings columns should be evaluated. It's significantly
# slower unfortunately due to design choices I made before coming up with an idea to test features.
def get_score_function(model, columns, constant_data, eval_embeds):
    def score(X, y):
        print("Scoring!")
        df = pd.DataFrame(X, columns=columns)
        constant_cols = constant_data.columns.tolist()

        # Check if embeddings are actually permuted
        embedding_permuted = not constant_data.equals(df[constant_cols])

        # This step is needed if embeddings are to be evaluated, and we need it before preprocessing
        permutation_map = None
        if eval_embeds and embedding_permuted:
            permutation_map = constant_data.set_index(constant_cols)
            permutation_map[constant_cols] = pd.DataFrame(df[constant_cols].to_numpy(), index=permutation_map.index)

        # Get the preprocessing columns constant
        df[constant_cols] = constant_data.to_numpy()

        # Slightly different approach to prediction to allow the trick
        test_processed = model.data_preprocessor.preprocess_test_dataset(df)

        if eval_embeds and embedding_permuted:
            # Dirty trick to permute embeddings columns, we use the map to fill different values after preprocessing
            for i in range(len(test_processed.train)):
                test_processed.train[i][-1, :3] = torch.from_numpy(
                    permutation_map.loc[tuple(test_processed.train[i][-1, :3].tolist())].to_numpy()
                )

        test_loader = model.get_test_loader(test_processed)

        y_pred = model.transform(test_loader)

        return root_mean_squared_error(y, y_pred.clip(0, 20))
    return score


if __name__ == "__main__":
    train_set_path = "data/processed_validation/train.parquet"
    test_set_path = "data/processed_validation/test.parquet"
    label_column = "item_cnt_month"
    # Whether you want to test on the test set or on the last month of training set
    use_train = True
    # Should embeddings be evaluated
    evaluate_embeddings = True

    # Do appropriate query if you want the train to go faster
    train = read_parquet(train_set_path)  # .query("date_block_num > 20").reset_index(drop=True)
    y_train = train[[label_column]]
    train.drop(label_column, axis=1, inplace=True)

    test = read_parquet(test_set_path)
    y_test = test[[label_column]]
    test.drop(label_column, axis=1, inplace=True)

    model = get_model(lstm_net)
    model.fit(train, y_train)

    if use_train:
        # First of all, cut of last observation from reference dict
        reference_dict = model.data_preprocessor.reference_dict
        for key in reference_dict.keys():
            reference_dict[key] = reference_dict[key][:-1]
        # Second, prepare train set appropriately
        max_dbn = train['date_block_num'].max()
        score_set = train.query('date_block_num == @max_dbn')
        score_y = y_train.loc[score_set.index]
    else:
        score_set, score_y = test, y_test

    cols = train.columns.tolist()

    # This will be used to keep the preprocessing right. I will then use some dirty tricks to permute embedding columns.
    constant_data = score_set[['item_id', 'item_category_id', 'shop_id']]

    score_func = get_score_function(model, cols, constant_data, evaluate_embeddings)

    base_score, score_decreases = get_score_importances(
           score_func, score_set.to_numpy(), score_y.to_numpy(), random_state=234234, n_iter=1
    )

    feature_importances = np.mean(score_decreases, axis=0)

    # We sort ascending because when score_decreases is negative, it means it increases,
    # which is what we care about (if RMSE increases it means the feature is important)
    result = sorted(list(zip(feature_importances, cols)), key=lambda x: x[0])

    for result_row in result:
        print(result_row)
