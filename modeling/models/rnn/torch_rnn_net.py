import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import *
from torch.utils.data import DataLoader
from modeling.models.base_model import Model
from modeling.models.rnn.dense_sets_holder import DenseSetsHolder
from modeling.models.rnn.rnn_data_preprocessor import RNNDataPreprocessor
from modeling.models.rnn.rnn_dataset import RNNDataset


class RNNNet(torch.nn.Module, Model):
    def __init__(
            self, items_path, items_categories_path, shops_path, average_dense_sets,
            input_features_num, num_epochs, batch_size, learning_rate,
            rnn_module, rnn_layers_num, rnn_input_dim, rnn_hidden_output_dim,
            initialize_memory_gate_bias,
            pre_rnn_layers_num, pre_rnn_dim,
            post_rnn_layers_num, post_rnn_dim, pre_output_dim,
            save_temporary_models, plot_training_history,
            device='cuda', seed=234534
    ):
        super(RNNNet, self).__init__()
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.save_temporary_models = save_temporary_models
        self.plot_training_history = plot_training_history

        self.items_path = items_path
        self.items_categories_path = items_categories_path
        self.shops_path = shops_path
        self.average_dense_sets = average_dense_sets

        self.data_preprocessor = None
        self.dense_sets_holder = None

        self.target_device = torch.device(device)

        self.rnn_layers_num = rnn_layers_num
        self.rnn_hidden_output_dim = rnn_hidden_output_dim
        self.is_lstm = rnn_module == torch.nn.LSTM

        # RNN training trick, reference https://danijar.com/tips-for-training-recurrent-neural-networks/
        self.hidden_learned = nn.Embedding(self.rnn_layers_num, self.rnn_hidden_output_dim)
        self.hidden_idx = torch.LongTensor(list(range(self.rnn_layers_num))).to(self.target_device, non_blocking=True)

        if self.is_lstm:
            self.cell_learned = nn.Embedding(self.rnn_layers_num, self.rnn_hidden_output_dim)

        pre_rnn_modules = []
        if pre_rnn_layers_num == 1:
            pre_rnn_modules.extend([nn.Linear(input_features_num, rnn_input_dim), nn.ReLU()])
        else:
            pre_rnn_modules.extend([nn.Linear(input_features_num, pre_rnn_dim), nn.ReLU()])
            for i in range(pre_rnn_layers_num - 2):
                pre_rnn_modules.extend([nn.Linear(pre_rnn_dim, pre_rnn_dim), nn.ReLU()])
            pre_rnn_modules.extend([nn.Linear(pre_rnn_dim, rnn_input_dim), nn.ReLU()])

        self.pre_rnn_layers = nn.Sequential(*pre_rnn_modules)

        self.recurrent_unit = rnn_module(
            rnn_input_dim, self.rnn_hidden_output_dim, batch_first=True, num_layers=self.rnn_layers_num
        )

        # RNN training trick, reference https://danijar.com/tips-for-training-recurrent-neural-networks/
        if initialize_memory_gate_bias:
            for names in self.recurrent_unit._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(self.recurrent_unit, name)
                    n = bias.size(0)
                    if type(self.recurrent_unit) is torch.nn.GRU:
                        start, end = 0, n // 3
                        bias.data[start:end].fill_(-1.)
                    elif self.is_lstm:
                        start, end = n//4, n//2
                        bias.data[start:end].fill_(1.)

        # I assumed post rnn FC modules are followed by the final module. This can be easily changed.
        post_rnn_modules = []
        if post_rnn_layers_num == 1:
            post_rnn_modules.extend([nn.Linear(rnn_hidden_output_dim, pre_output_dim), nn.ReLU()])
        else:
            post_rnn_modules.extend([nn.Linear(rnn_hidden_output_dim, post_rnn_dim), nn.ReLU()])
            for i in range(post_rnn_layers_num - 2):
                post_rnn_modules.extend([nn.Linear(post_rnn_dim, post_rnn_dim), nn.ReLU()])
            post_rnn_modules.extend([nn.Linear(post_rnn_dim, pre_output_dim), nn.ReLU()])
        post_rnn_modules.append(nn.Linear(pre_output_dim, 1))

        self.post_rnn_layers = nn.Sequential(*post_rnn_modules)

    def forward(self, inputs, lens):
        indices = torch.argsort(-lens)
        indices_back = torch.argsort(indices)

        # Removing first three indexing columns from the input
        inputs_sliced = inputs[:, :, 3:]

        # Pre RNN FC layers
        inputs_fc_pre = self.pre_rnn_layers(inputs_sliced)

        # Pack the sequence
        inputs_packed = pack_padded_sequence(inputs_fc_pre[indices], lens[indices], batch_first=True)

        hidden_state = self.get_learnable_hidden_state(inputs_sliced.shape[0])
        lstm_output, _ = self.recurrent_unit(inputs_packed, hidden_state)

        # Reverse operation, pad the packed sequence
        input_post_rnn, _ = pad_packed_sequence(lstm_output, batch_first=True)

        # Post RNN FC layers
        input_final = self.post_rnn_layers(F.relu(input_post_rnn))[indices_back]

        # Return with proper shape. It is needed to always keep the same shape, which is not the case when the batch
        # happens to be filled with sequences of unit length.
        return input_final.reshape(lens.shape[0], -1)

    def fit(self, train, y_train, valid_set_tuple=None):
        # Print all parameters so you're sure you got what you wanted.
        print(self)

        criterion = nn.MSELoss()

        print("Preparing datasets..")
        self.dense_sets_holder = DenseSetsHolder(
            self.items_path, self.items_categories_path, self.shops_path,
            ("item_id", train['item_id'].drop_duplicates()),
            ("item_category_id", train['item_category_id'].drop_duplicates()),
            ("shop_id", train['shop_id'].drop_duplicates())
        )

        self.data_preprocessor = RNNDataPreprocessor()
        preprocessed_train_data = self.data_preprocessor.preprocess_train_dataset(train, y_train)
        train_dataset = RNNDataset(*preprocessed_train_data, self.dense_sets_holder, self.average_dense_sets)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=3,
                                  collate_fn=self.zip_collate, pin_memory=True)

        valid_set_tuple_postproc = None

        if valid_set_tuple is not None:
            test, y_test = valid_set_tuple
            preprocessed_test_data = self.data_preprocessor.preprocess_test_dataset(test)
            test_dataset = RNNDataset(*preprocessed_test_data, self.dense_sets_holder, self.average_dense_sets)
            test_loader = self.get_test_loader(test_dataset)
            valid_set_tuple_postproc = (test_loader, y_test)

        print("Finished preparing datasets, let's train!")

        batches_num = len(train_dataset) / self.batch_size
        # ~ 4 reports per epoch
        batch_report_interval = batches_num // 4

        print("Number of batches for dataset: ", batches_num)

        self.zero_grad()
        self.train()
        self.to(self.target_device, non_blocking=True)

        optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)

        # For plotting and monitoring purposes
        x_losses = []
        train_losses = []
        test_losses = []
        min_test_loss = 999

        for epoch in range(self.num_epochs):
            running_loss = 0.0

            for i, data in enumerate(train_loader, 0):
                # Get the inputs
                inputs, lens, labels = data
                labels = labels.to(self.target_device, non_blocking=True)
                inputs = inputs.to(self.target_device, non_blocking=True)
                lens = lens.to(self.target_device, non_blocking=True)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self.forward(inputs, lens)

                # Learning on all time steps outputs - need a mask to extract the relevant ones
                labels_mask = (labels >= 0)

                outputs_extracted = outputs[labels_mask]

                # Get the RMSE
                loss = torch.sqrt(criterion(outputs_extracted, labels[labels_mask]))

                loss.backward()

                optimizer.step()

                # Gather statistics
                running_loss += loss.item()

                # Perform printing and saving/plotting after X batches
                if (i + 1) % batch_report_interval == 0:
                    x_losses.append('[%d, %d]' % (epoch + 1, i + 1))

                    train_loss = running_loss / batch_report_interval
                    train_losses.append(train_loss)
                    print('[%d, %5d] train-loss: %.3f' % (epoch + 1, i + 1, train_loss))

                    # Reset the loss monitor
                    running_loss = 0.0

                    if valid_set_tuple_postproc:
                        test_loader, y_test = valid_set_tuple_postproc
                        results = self.transform(test_loader, False, False)

                        test_loss = torch.sqrt(
                            F.mse_loss(results.clamp(0, 20), torch.Tensor(y_test.to_numpy()).squeeze())
                        ).item()

                        test_losses.append(test_loss)
                        print('[%d, %5d] test-loss: %.3f' % (epoch + 1, i + 1, test_loss))

                        if self.save_temporary_models and test_loss < min_test_loss:
                            min_test_loss = test_loss
                            torch.save(self.state_dict(), "model_{}_{}.pt".format(test_loss, epoch))

        print('Finished Training')

        # Plot the loss chart afterwards
        if self.plot_training_history:
            plt.figure()
            plt.plot(x_losses, train_losses)
            if valid_set_tuple_postproc:
                plt.plot(x_losses, test_losses)
            axes = plt.gca()
            axes.set_ylim([0.25, 1.5])
            fig = plt.gcf()
            fig.savefig("./plot_{}.png".format(time.time()))

    def transform(self, test, to_cpu=True, to_numpy=True):
        assert self.data_preprocessor is not None, \
            "Cannot predict without data preparation/fitting. Fit the model first."

        # Depending on whether it's a monitoring phase or an actual prediction, different preparation is needed
        if type(test) is DataLoader:
            test_loader = test
        else:
            test_data_preprocessed = self.data_preprocessor.preprocess_test_dataset(test)
            test_dataset = RNNDataset(*test_data_preprocessed, self.dense_sets_holder, self.average_dense_sets)
            test_loader = self.get_test_loader(test_dataset)

        self.eval()
        results = []
        with torch.no_grad():
            for inputs_part, lens_part, _ in test_loader:
                results_chunk = self.forward(
                    inputs_part.to(self.target_device, non_blocking=True),
                    lens_part.to(self.target_device, non_blocking=True)
                )
                # Getting last output from test prediction
                results.extend([x[l - 1] for x, l in zip(results_chunk, lens_part)])

            results_final = torch.Tensor(results).detach()

            if to_cpu:
                results_final = results_final.cpu()
            if to_numpy:
                results_final = results_final.numpy()

        self.train()

        return results_final

    def reshape_hidden_state(self, tensor, batch_len):
        # Strange way of reshaping the actual hidden state
        return tensor.unsqueeze(1).expand(self.rnn_layers_num, batch_len, self.rnn_hidden_output_dim).contiguous()

    def get_learnable_hidden_state(self, batch_len):
        hidden_state = self.reshape_hidden_state(self.hidden_learned(self.hidden_idx), batch_len)
        if self.is_lstm:
            cell_state = self.reshape_hidden_state(self.cell_learned(self.hidden_idx), batch_len)
            hidden_state = (hidden_state, cell_state)

        return hidden_state

    @staticmethod
    def zip_collate(batch):
        """
        Custom collate function for torch data loader.
        """
        inputs, lens, labels = zip(*batch)
        # Pad the sequences and labels to equal length.
        # If processing test set, all labels will be None, in that case no processing is done.
        if labels[0] is not None:
            labels = pad_sequence(labels, batch_first=True, padding_value=-999)
        else:
            labels = None
        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        lens = torch.LongTensor(lens)
        return inputs, lens, labels

    def get_test_loader(self, rnn_dataset):
        return DataLoader(
                rnn_dataset, batch_size=3200, shuffle=False, num_workers=3, collate_fn=self.zip_collate, pin_memory=True
            )
