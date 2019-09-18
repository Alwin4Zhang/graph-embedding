# -*-coding: utf-8 -*-
"""
    @project:PycharmProjects
    @author:alwin
    @file:node2vec.py
    @time:2019-09-08 17:13:30
    @github:alwin114@hotmail.com
"""
from gensim.models import Word2Vec
import pandas as pd
from .walker import RandomWalker
import logging
from os.path import join as join_path, dirname
import numpy as np


class Node2Vec(object):
    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1):
        self.graph = graph
        self._embedding = {}
        self.walker = RandomWalker(graph, p=p, q=q)
        print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks,
            walk_length=walk_length,
            workers=workers,
            verbose=1
        )

        self.sentences = list(filter(lambda x:len(x) > 3,self.sentences))
        with open('sentences.txt', 'w+') as f:
            sentences = [','.join([str(node) for node in sentence]) for sentence in self.sentences]
            f.write('\n'.join(sentences))
            print('sentence generate all done!!!')

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, filename='last_version', **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size 
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter
        save_file_name = filename

        print("Learning embedding vectors...")
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model
        model_save_path = join_path(dirname(dirname(__file__)), 'model_train/' + save_file_name)
        model.save(model_save_path)

        return model

    def get_embeddings(self, ):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            # self._embeddings[word] = self.w2v_model.wv[word]
            if word not in self.w2v_model:
                self._embeddings[word] = np.zeros(128)
            else:
                self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings
