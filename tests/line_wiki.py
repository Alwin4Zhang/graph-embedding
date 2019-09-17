# -*- coding: utf-8 -*-
'''
    @Time    : 2019/9/11 2:57 PM
    @Author  : alwin
    @Email   : alwin114@hotmail.com
'''

import numpy as np
import sys
sys.path.append('./')
from models.classify import read_node_label, Classifier
from models.line import LINE
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from os.path import join as join_path, dirname
from pprint import pprint


def evaluate_embeddings(embeddings):
    data_path = join_path(dirname(dirname(__file__)), 'data/bello_kg/graph_labels_last_version.txt')
    # data_path = join_path(dirname(dirname(__file__)), 'data/bello_kg/graph_labels_v1.1.txt')
    X, Y = read_node_label(data_path)
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings, ):
    data_path = join_path(dirname(dirname(__file__)), 'data/bello_kg/graph_labels_last_version.txt')
    # data_path = join_path(dirname(dirname(__file__)), 'data/bello_kg/graph_labels_v1.1.txt')
    X, Y = read_node_label(data_path)

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_path = join_path(dirname(dirname(__file__)), 'data/bello_kg/last_version_edgelist.txt')
    G = nx.read_edgelist(data_path,
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    pprint(G.edges(data=True))
    model = LINE(G, embedding_size=128, order='second')
    model.train(batch_size=1024, epochs=50, verbose=2)
    embeddings = model.get_embeddings()

    evaluate_embeddings(embeddings)
    plot_embeddings(embeddings)
