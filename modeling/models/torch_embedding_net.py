import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.autograd.variable import Variable
from torch.utils.data import TensorDataset, DataLoader
from modeling.models.base_model import Model


class EmbeddingNet(torch.nn.Module, Model):
    def __init__(
            self, items_path, items_categories_path, shops_path,
            non_embedding_features_num, num_epochs, batch_size
    ):
        super(EmbeddingNet, self).__init__()

        self.device = torch.device('cuda')
        self.cols_in_order = ['item_id', 'item_category_id', 'shop_id']
        self.cols_rest = None
        self.standardizer = StandardScaler()

        items = pd.read_csv(items_path)
        items_categories = pd.read_csv(items_categories_path)
        shops = pd.read_csv(shops_path)

        embeds_size = 20
        non_embedding_features_out = 196

        items_size = items.shape[0]
        items_categories_size = items_categories.shape[0]
        shops_size = shops.shape[0]

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.embedding_items = torch.nn.Embedding(items_size, embeds_size)
        self.embedding_categories = torch.nn.Embedding(items_categories_size, embeds_size)
        self.embedding_shops = torch.nn.Embedding(shops_size, embeds_size)
        self.input_rest = torch.nn.Linear(non_embedding_features_num, non_embedding_features_out)
        self.hidden_concat = torch.nn.Linear(embeds_size * 3 + non_embedding_features_out, 50)
        self.final = torch.nn.Linear(50, 1)

    def forward(self, input):
        item_indices = input[:, 0].long()
        categories_indices = input[:, 1].long()
        shops_indices = input[:, 2].long()
        rest = input[:, 3:]

        item_embeds = F.relu(self.embedding_items(item_indices))
        categories_embeds = F.relu(self.embedding_categories(categories_indices))
        shops_embeds = F.relu(self.embedding_shops(shops_indices))
        rest_out = F.relu(self.input_rest(rest))

        input_concat = torch.cat([item_embeds, categories_embeds, shops_embeds, rest_out], dim=1)
        input_concat = F.relu(self.hidden_concat(input_concat))
        input_concat = self.final(input_concat)

        return input_concat

    def preprocess_data(self, dataset):
        index_cols = dataset[self.cols_in_order].values
        normalized_cols = self.standardizer.transform(dataset[self.cols_rest].values)
        normalized = np.concatenate([index_cols, normalized_cols], axis=1)
        normalized_wo_nan = np.nan_to_num(normalized)
        return torch.from_numpy(normalized_wo_nan).float().to(self.device)

    def fit(self, train, y_train, valid_set_tuple=None):
        criterion = nn.MSELoss()

        batches_num = len(train) / self.batch_size
        # ~ 4 reports per epoch
        batch_report_interval = batches_num // 4

        self.cols_rest = sorted(list(set(train.columns.tolist()) - set(self.cols_in_order)))
        self.standardizer.fit(train[self.cols_rest].values)

        train_processed = self.preprocess_data(train)

        train_dataset = TensorDataset(train_processed, torch.from_numpy(y_train.values).to(self.device).float())

        loader_train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        valid_set_tuple_postproc = None
        if valid_set_tuple:
            test, y_test = valid_set_tuple
            valid_set_tuple_postproc = self.preprocess_data(test), torch.Tensor(y_test.values).squeeze()

        self.zero_grad()
        self.train()
        self.to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for i, data in enumerate(loader_train, 0):
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                inputs, labels = Variable(inputs.to(self.device)), Variable(labels.to(self.device))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs)

                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                # print statistics
                running_loss += loss.item()

                if (i + 1) % batch_report_interval == 0:
                    print('[%d, %5d] train-loss: %.3f' % (epoch + 1, i + 1, np.sqrt(running_loss / batch_report_interval)))

                    running_loss = 0.0

            if valid_set_tuple_postproc:
                test, y_test = valid_set_tuple_postproc
                results = self.transform(test, True, False)
                test_loss = torch.sqrt(F.mse_loss(results.clamp(0, 20), y_test)).item()
                print('[%d, %5d] test-loss: %.3f' % (epoch + 1, i + 1, test_loss))

        print('Finished Training')

    def transform(self, test, to_cpu=True, to_numpy=True):
        # Depending on whether it's a monitoring phase or an actual prediction, different preparation is needed
        if type(test) is not torch.Tensor:
            test = self.preprocess_data(test)

        test_loader = DataLoader(test, batch_size=6400, shuffle=False, num_workers=0)

        self.eval()
        result = []
        for chunk in test_loader:
            result.append(self.forward(chunk).squeeze().detach())

        result = torch.cat(result)

        if to_cpu:
            result = result.cpu()
        if to_numpy:
            result = result.numpy()

        self.train()
        return result
