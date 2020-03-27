from abc import ABC, abstractmethod
import re
import pandas as pd
import numpy as np


class SentencesProcessorIterator:
    def __init__(self, sentences):
        self.regexp = re.compile('[\W_]+', re.UNICODE)
        self.sentences = sentences

    def __iter__(self):
        for sentence in self.sentences:
            yield self.regexp.sub(' ', sentence.lower()).split()


class DenseDatasetGenerator(ABC):
    """
    Base class for dense datasets generators.
    """

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
        sentence_vectors[id_attr] = dataset[id_attr].to_numpy()

        return sentence_vectors

    @abstractmethod
    def _get_vectors_container(self, processed_sentences):
        pass

    def _get_mean_vector(self, vectors_container):
        return np.array(list(vectors_container.values()), dtype='float32').mean(axis=0)

    @staticmethod
    def _extract_processed_sentences(dataset, sentence_attr):
        raw_sentences = dataset[sentence_attr].tolist()
        return SentencesProcessorIterator(raw_sentences)


class DenseDatasetGeneratorParametrized(DenseDatasetGenerator, ABC):
    def __init__(self, embedding_size, num_epochs, learning_rate):
        super(DenseDatasetGeneratorParametrized, self).__init__()
        self.embedding_size = embedding_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
