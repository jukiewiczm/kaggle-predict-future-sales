from modeling.models.torch_embedding_net import EmbeddingNet
from modeling.models.rnn.torch_rnn_net import RNNNet
from modeling.models.xgb_wrapper import XGBWrapper
import torch.nn


def get_model(config):
    return config['model'](**config['model_params'])


simple_xgb = {'model': XGBWrapper, 'model_params': {'n_jobs': 8}}
# The one below requires py-xgboost-gpu package, which was causing some problems during installation, so it's not
# listed by default in environment.yml.
cuda_xgb = {'model': XGBWrapper, 'model_params': {'n_jobs': 8, 'tree_method': 'gpu_hist'}}
pytorch_embed = {'model': EmbeddingNet,
                 'model_params': {'items_path': "data/items.csv",
                                  'items_categories_path': "data/item_categories.csv",
                                  'shops_path': "data/shops.csv",
                                  'num_epochs': 2,
                                  'batch_size': 128,
                                  'non_embedding_features_num': 43
                                  }
                 }
lstm_net = {'model': RNNNet,
            'model_params': {
                    'items_path': "data/starspace_items_dense.parquet",
                    'items_categories_path': "data/starspace_items_categories_dense.parquet",
                    'shops_path': "data/starspace_shops_dense.parquet",
                    'average_dense_sets': False,
                    'input_features_num': 193, 'num_epochs': 1, 'batch_size': 256, "learning_rate": 1e-5,
                    'rnn_module': torch.nn.GRU, 'rnn_layers_num': 1, 'rnn_input_dim': 188, 'rnn_hidden_output_dim': 188,
                    'initialize_memory_gate_bias': True,
                    'pre_rnn_layers_num': 2, 'pre_rnn_dim': 376,
                    'post_rnn_layers_num': 2, 'post_rnn_dim': 188, 'pre_output_dim': 94,
                    'save_temporary_models': False,
                    'plot_training_history': True
                }
            }