#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
def lda_gen(vocabulary, alpha, beta, xi):
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

