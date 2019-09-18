# -*-coding: utf-8 -*-
"""
    @project:PycharmProjects
    @author:alwin
    @file:node2vec_wiki.py
    @time:2019-09-08 22:56:28
    @github:alwin114@hotmail.com
"""

import numpy as np
import sys

sys.path.append("./")

from models.classify import read_node_label, Classifier
from models.node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from os.path import join as join_path, dirname
from pprint import pprint


def evaluate_embeddings(embeddings):
    # data_path = join_path(dirname(dirname(__file__)),'data/wiki/wiki_labels.txt')
    data_path = join_path(dirname(dirname(__file__)), 'data/bello_kg/graph_labels_last_version.txt')
    data_path = join_path(dirname(dirname(__file__)), 'data/bello_kg/graph_labels_v1.1.txt')
    X, Y = read_node_label(data_path)
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings, ):
    # data_path = join_path(dirname(dirname(__file__)),'data/wiki/wiki_labels.txt')
    data_path = join_path(dirname(dirname(__file__)), 'data/bello_kg/graph_labels_last_version.txt')
    data_path = join_path(dirname(dirname(__file__)), 'data/bello_kg/graph_labels_v1.1.txt')
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
    # data_path = join_path(dirname(dirname(__file__)),'data/wiki/Wiki_edgelist.txt')
    data_path = join_path(dirname(dirname(__file__)), 'data/bello_kg/last_version_edgelist.txt')
    data_path = join_path(dirname(dirname(__file__)), 'data/bello_kg/kg_v1.1_edgelist.txt')
    G = nx.read_edgelist(data_path,create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    print('Graph generate successed!!!')
    # pprint(G.edges(data=True))
    model = Node2Vec(G, walk_length=10, num_walks=80,
                     p=0.25, q=4, workers=1)
    model.train(window_size=5, iter=3, filename='this_version')
    embeddings = model.get_embeddings()
    print('model generate successed!!!')
    evaluate_embeddings(embeddings)
    plot_embeddings(embeddings)


"""co_freq > 5
1.embedding_size = 128
2.embedding_size = 64
"""

{'micro': 0.5120481927710844, 'macro': 0.3160060129569028, 'samples': 0.5120481927710844, 'weighted': 0.46925516825524244, 'acc': 0.5120481927710844}
{'micro': 0.516566265060241, 'macro': 0.2328202972088272, 'samples': 0.516566265060241, 'weighted': 0.4567222038874553, 'acc': 0.516566265060241}

{'micro': 0.5128012048192772, 'macro': 0.3162647657487756, 'samples': 0.5128012048192772, 'weighted': 0.4705223929488406, 'acc': 0.5128012048192772}
{'micro': 0.5075301204819277, 'macro': 0.2146511487393228, 'samples': 0.5075301204819277, 'weighted': 0.4429392835292297, 'acc': 0.5075301204819277}

"""co_freq > 10
1.embedding_size = 128
2.embedding_size = 64
"""

{'micro': 0.5380952380952381, 'macro': 0.28223291770351633, 'samples': 0.5380952380952381, 'weighted': 0.49883743906891853, 'acc': 0.5380952380952381}
{'micro': 0.5416666666666666, 'macro': 0.23771227924417118, 'samples': 0.5416666666666666, 'weighted': 0.49713215933462646, 'acc': 0.5416666666666666}


"""
kg_v2
co_freq > 10
1.embedding_size = 128
2.embedding_size = 64
"""
# last_version 
last_version_results = [
    {'micro': 0.525323910482921, 'macro': 0.27683250281179533, 'samples': 0.525323910482921, 'weighted': 0.48497018963059807, 'acc': 0.525323910482921},
    {'micro': 0.5276796230859835, 'macro': 0.29153499448520087, 'samples': 0.5276796230859835, 'weighted': 0.4903638663058918, 'acc': 0.5276796230859835},
    {'micro': 0.5300353356890459, 'macro': 0.3049213204017436, 'samples': 0.5300353356890459, 'weighted': 0.49507437966367557, 'acc': 0.5300353356890459},
    {'micro': 0.5206124852767963, 'macro': 0.27416352628266843, 'samples': 0.5206124852767963, 'weighted': 0.4809552508963732, 'acc': 0.5206124852767963},
    {'micro': 0.519434628975265, 'macro': 0.27738824751545393, 'samples': 0.519434628975265, 'weighted': 0.4791390969170188, 'acc': 0.519434628975265},

    # filter sentence that length < 3
    {'micro': 0.5429917550058893, 'macro': 0.31531751926799834, 'samples': 0.5429917550058893, 'weighted': 0.5183483357562076, 'acc': 0.5429917550058893},
    # not filter short sentence 
    {'micro': 0.6336866902237926, 'macro': 0.4199216176569023, 'samples': 0.6336866902237926, 'weighted': 0.6109793646140534, 'acc': 0.6336866902237926}
]

# this_version
this_version_results = [
    {'micro': 0.574793875147232, 'macro': 0.23419897161027578, 'samples': 0.574793875147232, 'weighted': 0.5200289043723975, 'acc': 0.574793875147232},
    {'micro': 0.5547703180212014, 'macro': 0.20998177396040413, 'samples': 0.5547703180212014, 'weighted': 0.4938178281706436, 'acc': 0.5547703180212014},
    {'micro': 0.5606595995288575, 'macro': 0.22172107279006026, 'samples': 0.5606595995288575, 'weighted': 0.5046892001870162, 'acc': 0.5606595995288575},
    {'micro': 0.5653710247349824, 'macro': 0.21484777861080412, 'samples': 0.5653710247349824, 'weighted': 0.5038664905740498, 'acc': 0.5653710247349824},
    {'micro': 0.5571260306242638, 'macro': 0.21823788786883944, 'samples': 0.5571260306242638, 'weighted': 0.5018728296025498, 'acc': 0.5571260306242638},
    
    # filter sentence that length < 3
    {'micro': 0.5712603062426383, 'macro': 0.23315737886509924, 'samples': 0.5712603062426383, 'weighted': 0.5215596291112815, 'acc': 0.5712603062426383},
    # not filter short sentence 
    {'micro': 0.6065959952885748, 'macro': 0.28232299330078087, 'samples': 0.6065959952885748, 'weighted': 0.5569100402742883, 'acc': 0.6065959952885748}
]