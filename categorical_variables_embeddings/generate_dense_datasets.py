import logging
import re

import gensim
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class SentencesProcessorIterator:
    def __init__(self, sentences):
        self.regexp = re.compile('[\W_]+', re.UNICODE)
        self.sentences = sentences

    def __iter__(self):
        for sentence in self.sentences:
            yield self.regexp.sub(' ', sentence.lower()).split()


class DenseDatasetGenerator(ABC):

    def transform(self, dataset, id_attr, sentence_attr):
        # fit_transform would indicate it's always trained on the dataset given which is not always the case
        """
        May perform fitting during the transformation.
        """
        processed_sentences = self._extract_processed_sentences(dataset, sentence_attr)
        vectors_container = self._get_vectors_container(processed_sentences)
        mean_vec = self._get_mean_vector(vectors_container)

        sentence_vectors = []
        for words in processed_sentences:
            word_vecs = []
            for w in words:
                if w in vectors_container:
                    word_vecs.append(vectors_container[w])

            if len(word_vecs) > 0:
                sentence_vectors.append(np.array(word_vecs).mean(axis=0))
            else:
                sentence_vectors.append(mean_vec)

        sentence_vectors = pd.DataFrame(np.array(sentence_vectors))
        sentence_vectors.columns = ["{}_{}".format(id_attr, c) for c in sentence_vectors.columns]
        sentence_vectors[id_attr] = dataset[id_attr].values

        return sentence_vectors

    @abstractmethod
    def _get_vectors_container(self, processed_sentences):
        pass

    @abstractmethod
    def _get_mean_vector(self, vectors_container):
        pass

    def _extract_processed_sentences(self, dataset, sentence_attr):
        raw_sentences = dataset[sentence_attr].tolist()
        return SentencesProcessorIterator(raw_sentences)


class GensimDenseGenerator(DenseDatasetGenerator):

    def _get_vectors_container(self, processed_sentences):
        model = gensim.models.Word2Vec(processed_sentences, min_count=1, size=100, iter=20, batch_words=100)
        return model.wv

    def _get_mean_vector(self, vectors_container):
        return vectors_container.vectors.mean(axis=0)


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

    def _get_mean_vector(self, vectors_container):
        return np.array(list(self.word_vecs_dict.values()), dtype='float32').mean(axis=0)


def produce_dense_dataset(generator, input_path, id_attr_name, sentence_attr_name, output_path):
    dataset = pd.read_csv(input_path)
    dataset_dense = generator.transform(dataset, id_attr_name, sentence_attr_name)
    dataset_dense.to_parquet(output_path)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # *** Choose your generator
    # generator = GensimDenseGenerator()
    generator = FileBasedDenseGenerator("data/wiki.ru.vec")
    # ***

    produce_dense_dataset(generator, "data/items.csv", 'item_id', 'item_name', "data/gensim_items_dense.parquet")
    produce_dense_dataset(generator, "data/item_categories.csv", 'item_category_id', 'item_category_name',
                          "data/gensim_items_categories_dense.parquet")
    produce_dense_dataset(generator, "data/shops.csv", 'shop_id', 'shop_name', "data/gensim_shops_dense.parquet")
