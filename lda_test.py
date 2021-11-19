"""Latent Dirichlet Allocation

Patrick Wang, 2021
"""
from typing import List

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import numpy as np


def lda_gen(vocabulary: List[str], alpha: np.ndarray, beta: np.ndarray, xi: int) -> List[str]:
    '''
    vocabulary: list of strings(length V)
    alpha: topic distribution parameter vector, numpy array of size (k,)
    beta: topic-word matrix, numpy array of size (k, V)
    xi: Poisson parameter (scalar) for document size distribution
    returns words: list of words (strings) in a document
    '''
    topic_proportion = np.random.dirichlet(alpha)
    word_topic = np.random.choice(range(np.shape(beta)[0]), np.random.poisson(xi), p=topic_proportion)
    words = []
    for topic in word_topic:
        words.append(np.random.choice(vocabulary, p=beta[topic]))
    return words


def test():
    """Test the LDA generator."""
    vocabulary = [
        "bass", "pike", "deep", "tuba", "horn", "catapult",
    ]
    beta = np.array([
        [0.4, 0.4, 0.2, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.1, 0.0, 0.3, 0.3],
        [0.3, 0.0, 0.2, 0.3, 0.2, 0.0]
    ])
    alpha = np.array([0.2, 0.2, 0.2])
    xi = 50
    documents = [
        lda_gen(vocabulary, alpha, beta, xi)
        for _ in range(100)
    ]

    # Create a corpus from a list of texts
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]
    model = LdaModel(
        corpus,
        id2word=dictionary,
        num_topics=3,
    )
    print(model.alpha)
    print(model.show_topics())


if __name__ == "__main__":
    test()
