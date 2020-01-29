import logging
import os

import gensim
import pandas as pd
import numpy as np

from categorical_variables_embeddings.dense_dataset_generator import DenseDatasetGenerator, \
    DenseDatasetGeneratorParametrized

# In order to use starspace, you need to install it using bin/install_starspace.sh.
# Unfortunately, it won't be visible in an IDE. If you're not going to use starspace, just comment this import.
import starwrap as sw


class GensimDenseGenerator(DenseDatasetGeneratorParametrized):
    def _get_vectors_container(self, processed_sentences):
        model = gensim.models.Word2Vec(processed_sentences,
                                       min_count=1, size=self.embedding_size,
                                       iter=self.num_epochs, alpha=self.learning_rate)
        return model.wv

    def _get_mean_vector(self, vectors_container):
        return vectors_container.vectors.mean(axis=0)


class StarspaceDenseGenerator(DenseDatasetGeneratorParametrized):
    def _get_vectors_container(self, processed_sentences):
        concatenated = [" ".join(s) + f" __label__{i}\n" for i, s in enumerate(processed_sentences)]

        with open("tmp_sentences.txt", "w") as f_out:
            f_out.writelines(concatenated)

        arg = sw.args()
        arg.trainFile = "tmp_sentences.txt"
        arg.trainMode = 0
        arg.lr = self.learning_rate
        arg.epoch = self.num_epochs
        arg.dim = self.embedding_size
        arg.similarity = "dot"
        sp = sw.starSpace(arg)
        sp.init()
        sp.train()
        sp.saveModelTsv("tmp_embeds.tsv")

        vectors_container = {}
        with open("tmp_embeds.tsv", "r") as f_in:
            for line in f_in:
                if "__label__" not in line:
                    split_line = line.split()
                    key = split_line[0]
                    embedding = np.array([float(num) for num in split_line[1:]])
                    vectors_container[key] = embedding

        os.remove("tmp_sentences.txt")
        os.remove("tmp_embeds.tsv")

        return vectors_container


class FileBasedDenseGenerator(DenseDatasetGenerator):
    def __init__(self, file_path):
        self.fitted = False
        self._file_path = file_path
        self.word_vecs_dict = None

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, file_path):
        if self.file_path != file_path:
            self.fitted = False

        self.file_path = file_path

    def _get_vectors_container(self, processed_sentences):
        if not self.fitted:
            all_words = set(w for word in processed_sentences for w in word)
            word_vecs_dict = dict()

            with open(self.file_path) as f:
                next(f)  # Skip the header
                for line in f:
                    line_split = line.split()
                    # It looks like this because some words are "multi words" (with spaces) for some reason
                    word = " ".join(line_split[:len(line_split) - 300])

                    if word in all_words:
                        word_vec = np.array(line_split[-300:], dtype='float32')
                        word_vecs_dict[word] = word_vec

            self.fitted = True
            self.word_vecs_dict = word_vecs_dict
            logging.info("Done creating dict from file.")
        return self.word_vecs_dict


def produce_dense_dataset(generator, input_path, id_attr_name, sentence_attr_name, output_path):
    dataset = pd.read_csv(input_path)
    dataset_dense = generator.transform(dataset, id_attr_name, sentence_attr_name)
    dataset_dense.to_parquet(output_path)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # *** Choose your generator
    # generator = GensimDenseGenerator()
    # generator = FileBasedDenseGenerator("data/wiki.ru.vec")
    generator = StarspaceDenseGenerator()
    # ***

    produce_dense_dataset(generator, "data/items.csv", 'item_id', 'item_name', "data/starspace_items_dense.parquet")
    produce_dense_dataset(generator, "data/item_categories.csv", 'item_category_id', 'item_category_name',
                          "data/starspace_items_categories_dense.parquet")
    produce_dense_dataset(generator, "data/shops.csv", 'shop_id', 'shop_name', "data/starspace_shops_dense.parquet")
