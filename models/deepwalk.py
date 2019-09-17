# -*-coding: utf-8 -*-
"""
    @project:PycharmProjects
    @author:alwin
    @file:deepwalk.py
    @time:2019-09-07 21:49:52
    @github:alwin114@hotmail.com
"""
from .walker import RandomWalker
from gensim.models import Word2Vec
import pandas as pd
import logging


class DeepWalk(object):
    def __init__(self, graph, walk_length, num_walks, workers=1):
        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}
        self.walker = RandomWalker(graph, p=1, q=1)
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks,
            walk_length=walk_length,
            workers=workers,
            verbose=1)
        print('sentences generate successed!!!')

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        kwargs['sentences'] = self.sentences
        kwargs['min_count'] = kwargs.get('min_count', 0)
        kwargs['size'] = embed_size
        kwargs['sg'] = 1  # skip_gram
        kwargs['hs'] = 1  # deepwalk use Hierarchical softmax
        kwargs['workers'] = workers
        kwargs['window'] = window_size
        kwargs['iter'] = iter 
        print('Learning embedding vectors...')
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = Word2Vec(**kwargs)
        print('Learning embedding vectors done!')

        self.w2v_model = model
        model.save('w2v_kg_v1.model')
        return model

    def get_embeddings(self, ):
        if self.w2v_model is None:
            print('model not train')
            return {}
        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]
        return self._embeddings
