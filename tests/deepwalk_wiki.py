# -*-coding: utf-8 -*-
"""
    @project:PycharmProjects
    @author:alwin
    @file:deepwalk_wiki.py
    @time:2019-09-07 22:32:29
    @github:alwin114@hotmail.com
"""

import numpy as np
import sys
sys.path.append("./") 
from os.path import join as join_path,dirname
from models.classify import read_node_label, Classifier
from models.deepwalk import DeepWalk
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE


def evaluate_embeddings(embeddings):
    # data_path = '../data/wiki/wiki_labels.txt'
    # data_path = join_path(dirname(dirname(__file__)),'data/bello_kg/graph_labels_last_version.txt')
    data_path = join_path(dirname(dirname(__file__)),'data/bello_kg/graph_labels_v1.1.txt')
    
    X, Y = read_node_label(data_path)
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings, ):
    # data_path = '../data/wiki/wiki_labels.txt'
    # data_path = join_path(dirname(dirname(__file__)),'data/bello_kg/graph_labels_last_version.txt')
    data_path = join_path(dirname(dirname(__file__)),'data/bello_kg/graph_labels_v1.1.txt')
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
    # data_path = '../data/wiki/Wiki_edgelist.txt'
    data_path = join_path(dirname(dirname(__file__)),'data/bello_kg/kg_v1.1_edgelist.txt')
    G = nx.read_edgelist(data_path,
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    print('Graph generate successed!!!')

    model = DeepWalk(G, walk_length=10, num_walks=50, workers=2)
    print('model generate successed!!!')
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()

    evaluate_embeddings(embeddings)
    plot_embeddings(embeddings)
