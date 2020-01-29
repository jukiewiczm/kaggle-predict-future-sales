import torch
import torch.nn

from categorical_variables_embeddings.generate_dense_datasets import StarspaceDenseGenerator, GensimDenseGenerator, \
    FileBasedDenseGenerator, produce_dense_dataset
from modeling.model_validation import StandardValidation
from modeling.models.rnn.torch_rnn_net import RNNNet
from modeling.utils import read_parquet

project_root_path = "kaggle-predict-future-sales"
target_col = "item_cnt_month"

prepare_submission = 0  # Change to 1 if you want to re-run guild job with tuned parameters.
test_reference = None
submission_path = "submissions/submission.csv.gz"
submission_path_absolute = f"{project_root_path}/{submission_path}"

if prepare_submission:
    train_set = read_parquet(f"{project_root_path}/data/processed_full/train.parquet")
    test_set = read_parquet(f"{project_root_path}/data/processed_full/test.parquet")
    test_reference = test_set[['ID']].astype("int32")
    test_set.drop([target_col, 'ID'], axis=1, inplace=True, errors='ignore')
    test_set = test_set[train_set.columns.drop(target_col)]
else:
    train_set_path = f"{project_root_path}/data/processed_validation/train.parquet"
    # It seems that taking it year back is a pretty good validation scheme.
    train_set = read_parquet(train_set_path).query("date_block_num < 22")
    test_set = read_parquet(train_set_path).query("date_block_num == 22")

# Actual tuning parameters
embedding_type = "starspace"
embedding_size = 50
average_dense_sets = 0
# A small workaround for guild, there might be a need to overwrite some values and it's the only way it works.
embedding_size_final = embedding_size
average_dense_sets_final = average_dense_sets

num_epochs = 1
batch_size = 256
learning_rate = 1e-5

pre_rnn_layers_num = 2
pre_rnn_dim = 376

rnn_module = "gru"
rnn_layers_num = 1
rnn_input_dim = 188
rnn_hidden_output_dim = 188
rnn_initialize_memory_gate_bias = 1

post_rnn_layers_num = 2
post_rnn_dim = 188
pre_output_dim = 94

# Preparations
## Embeddings
embeds_generator = {
    "starspace": StarspaceDenseGenerator(embedding_size_final),
    "gensim": GensimDenseGenerator(embedding_size_final),
    "wiki": FileBasedDenseGenerator(f"{project_root_path}/data/wiki.ru.vec")
}
generator = embeds_generator[embedding_type]

if embedding_type == "wiki":
    embedding_size_final = 300
    average_dense_sets_final = 1  # wiki embeddings are too big to concatenate them

produce_dense_dataset(generator, f"{project_root_path}/data/items.csv", 'item_id', 'item_name',
                      f"{project_root_path}/data/guild_item_embeddings.parquet")
produce_dense_dataset(generator, f"{project_root_path}/data/item_categories.csv", 'item_category_id', 'item_category_name',
                      f"{project_root_path}/data/guild_item_category_embeddings.parquet")
produce_dense_dataset(generator, f"{project_root_path}/data/shops.csv", 'shop_id', 'shop_name',
                      f"{project_root_path}/data/guild_shop_embeddings.parquet")

## RNN modules
rnn_modules = {"gru": torch.nn.GRU, "lstm": torch.nn.LSTM, "rnn": torch.nn.RNN}
rnn_module_class = rnn_modules[rnn_module]

if average_dense_sets_final:
    input_features_num = 43 + embedding_size_final
else:  # Dense sets concatenated
    input_features_num = 43 + 3 * embedding_size_final

# Get the model
model = RNNNet(
    f"{project_root_path}/data/guild_item_embeddings.parquet",
    f"{project_root_path}/data/guild_item_category_embeddings.parquet",
    f"{project_root_path}/data/guild_shop_embeddings.parquet",
    average_dense_sets_final,
    input_features_num,
    num_epochs, int(batch_size), learning_rate,
    rnn_module_class, rnn_layers_num, rnn_input_dim, rnn_hidden_output_dim, rnn_initialize_memory_gate_bias,
    pre_rnn_layers_num, pre_rnn_dim, post_rnn_layers_num, post_rnn_dim, pre_output_dim,
    save_temporary_models=False, plot_training_history=False
)

if prepare_submission:
    predictions = model.transform(test_set).clip(0, 20)
    test_reference['item_cnt_month'] = predictions
    test_reference.sort_values('ID').to_csv(submission_path_absolute, index=False, header=True, compression='gzip')
else:
    # Validate, save model (just in case), print output
    validation = StandardValidation(target_col, model)
    validation_result = validation.run(train_set, test_set, inner_train_validation=False, copy=False)

    torch.save(validation.model.state_dict(), f"{project_root_path}/model_{validation_result}.pt")

    print(f"loss: {validation_result}")
