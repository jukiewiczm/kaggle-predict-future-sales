from modeling.models.model_configs import *
from modeling.utils import read_parquet
import torch

# Script for preparing a competition submission

if __name__ == "__main__":
    train_set_path = "data/processed_full/train.parquet"
    test_set_path = "data/processed_full/test.parquet"
    target_col = "item_cnt_month"
    submission_path = "submissions/submission.csv.gz"

    # If you saved the state dict of the model at some point, you can load it instead of retraining the model
    model_state_dict_path = None

    train = read_parquet(train_set_path)
    y_train = train[[target_col]]
    train.drop(target_col, axis=1, inplace=True)

    test = read_parquet(test_set_path)
    test_reference = test[['ID']].astype("int32")
    test.drop([target_col, 'ID'], axis=1, inplace=True, errors='ignore')
    # Arrange the columns in train and test just in case
    test = test[train.columns]

    model = get_model(lstm_net)

    if model_state_dict_path:
        # Small trick here for loading trained model. Call the fit function with 0 epochs, so the preprocessing and
        # setup gets right, but no training occurs. Then load trained weights. This applies to RNN net only.
        if type(model) == RNNNet:
            model.num_epochs = 0
            model.fit(train, y_train)
        model.load_state_dict(torch.load(model_state_dict_path))
    else:
        model.fit(train, y_train)

    predictions = model.transform(test).clip(0, 20)
    test_reference['item_cnt_month'] = predictions
    test_reference.sort_values('ID').to_csv(submission_path, index=False, header=True, compression='gzip')
