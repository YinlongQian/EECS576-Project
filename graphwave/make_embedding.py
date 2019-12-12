import networkx as nx 
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import graphwave
from graphwave.shapes import build_graph
from graphwave.graphwave import *
import pickle

import random
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import seaborn as sns
sns.set_style('darkgrid')

def edge_features(node_emb_1, node_emb_2, operator):
    
    # combine two nodes' embeddings with specificed operator
    if operator == 'Average':
        edge = [((x + y) / 2.0) for x,y in zip(node_emb_1, node_emb_2)]
    elif operator == 'Hadamard':
        edge = [(x * y) for x,y in zip(node_emb_1, node_emb_2)]
    elif operator == 'Weighted-L1':
        edge = [abs(x - y) for x,y in zip(node_emb_1, node_emb_2)]
    elif operator == 'Weighted-L2':
        edge = [abs(x - y)**2 for x,y in zip(node_emb_1, node_emb_2)]
    else:
        print("Generate edge features: Operator not supported")
        print("Use default operator: Weighted-L1")
        edge = [abs(x - y) for x,y in zip(node_emb_1, node_emb_2)]
        
    return edge
def generate_edge_features(edge_list, node_embeddings, operator):
    edge_features_mtx = []
    
    # generate features for each edge in the list
    for node_index_1, node_index_2 in edge_list:
        node_emb_1 = node_embeddings[node_index_1]
        node_emb_2 = node_embeddings[node_index_2]
        
        edge_features_mtx.append(edge_features(node_emb_1, node_emb_2, operator))
        
    return edge_features_mtx

def generate_train_set(graph_train, num_edge_sample, node_embeddings, edge_operator,):
    edge_list = list(graph_train.edges)
    num_nodes = graph_train.number_of_nodes()
    
    train_edges = []
    train_edges_labels = [1] * num_edge_sample + [0] * num_edge_sample
    
    random.seed(0)
    
    # sample edges with label 1 (true edges)
    for edge_num in range(num_edge_sample):
        rand_index = random.randint(0, len(edge_list) - 1)
        
        #train_edges.append(tuple(edge_list[rand_index]))
        train_edges.append(edge_list[rand_index])
    non_edge_num = 0
    
    # sample edges with label 0 (non-exist edges)
    while(non_edge_num < num_edge_sample):
        rand_nodes = tuple(np.random.randint(low=0,high=num_nodes, size=2))
        
        if rand_nodes not in edge_list:
            train_edges.append(rand_nodes)
            non_edge_num += 1

    train_edges_features_mtx = generate_edge_features(train_edges, node_embeddings, edge_operator)
            
    return train_edges, train_edges_features_mtx, train_edges_labels

def generate_test_set(graph_test, node_embeddings, edge_operator):
    edge_list = graph_test.edges
    nodes_with_edge = set()
    
    for edge in edge_list:
        nodes_with_edge.add(edge[0])
        nodes_with_edge.add(edge[1])
    
    num_nodes = graph_test.number_of_nodes()
    
    test_edges = []
    test_edges_labels = []
    
    num_edge_sample = len(edge_list)
    non_edge_num = 0 
    # sample edges with label 0 (non-exist edges)
    
    while(non_edge_num < num_edge_sample):
        rand_nodes = tuple(np.random.randint(low=0,high=num_nodes, size=2))
        
        if rand_nodes not in edge_list:
            test_edges.append(rand_nodes)
            test_edges_labels.append(0)
            non_edge_num += 1
        
    for edge in edge_list:
        test_edges.append(edge)
        test_edges_labels.append(1)
    
    test_edges_features_mtx = generate_edge_features(test_edges, node_embeddings, edge_operator)
    
    return test_edges, test_edges_features_mtx, test_edges_labels


def build_clf(feature_mtx, response_vec):
   
    logistic_regression_model = LogisticRegression(random_state = 0,max_iter=5000,solver='liblinear',verbose=1,tol=1e-6)
    binary_clf = logistic_regression_model.fit(feature_mtx, response_vec)
    
    return binary_clf

def pred_links(feature_mtx, LR_clf):
    predict_edges_labels = LR_clf.predict(feature_mtx)
    
    return predict_edges_labels

def precision_recall(predict_labels, true_labels):
    
    cm = metrics.confusion_matrix(true_labels, predict_labels)
    print(cm)
    print(metrics.classification_report(true_labels, predict_labels))
    map = metrics.average_precision_score(true_labels, predict_labels)
    print('Mean Average Precision: {}'.format(map))
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, predict_labels)
    roc_auc = metrics.auc(fpr, tpr)
    print('Area Under ROC Curve: {}'.format(roc_auc))
    
    return 

def tester(chi_list,num_edge_samples= 400):
    
    #T-1 Only
    
    print("-----------------------BEGIN T-1-----------------------------")
    for edge_operator in ['Average','Hadamard','Weighted-L1','Weighted-L2']:
        train_edges, train_edges_features_mtx, train_edges_labels = generate_train_set(graph_train, num_edge_sample, chi_list[-2], edge_operator)
        test_edges, test_edges_features_mtx, test_edges_labels = generate_test_set(graph_test, chi_list[-1], edge_operator)

        LR_clf = build_clf(train_edges_features_mtx, train_edges_labels)

        print("Edge Operator: {}".format(edge_operator))
        predict_edges_labels = pred_links(test_edges_features_mtx, LR_clf)
        precision_recall(list(predict_edges_labels), list(test_edges_labels))
        print()
    print("------------------------END T-1----------------------------")
    
    print("-----------------------BEGIN SUM -----------------------------")

    prev_embedding = np.sum(np.asarray(chi_list[0:-1]),axis=0)
    cur_embedding = np.sum(np.asarray(chi_list),axis=0)    
        
    for edge_operator in ['Average','Hadamard','Weighted-L1','Weighted-L2']:
        train_edges, train_edges_features_mtx, train_edges_labels = generate_train_set(graph_train, num_edge_sample, prev_embedding, edge_operator)
        test_edges, test_edges_features_mtx, test_edges_labels = generate_test_set(graph_test, cur_embedding, edge_operator)

        LR_clf = build_clf(train_edges_features_mtx, train_edges_labels)

        print("Edge Operator: {}".format(edge_operator))
        predict_edges_labels = pred_links(test_edges_features_mtx, LR_clf)
        precision_recall(list(predict_edges_labels), list(test_edges_labels))
    print("-----------------------END SUM -----------------------------")

    embeddings = chi_list
    print("-----------------------BEGIN EXP DECAY ---------------------")
    for decay in [1,0.9,0.5,0.3]:
        print("------------ BEGIN: {} ---------------".format(decay))
        exps = [math.pow(math.e , (-i * decay)) for i in range(1,len(embeddings[:-2]))]
        exps.reverse()
        temp_embedding = np.zeros((embeddings[0]).shape) 
        for c,e in zip(embeddings[0:-2],exps):
             temp_embedding += e * c 
        prev_embedding = temp_embedding + embeddings[-2]

        # this is done so the last embedding has weight one. 
        cur_embedding = temp_embedding + exps[-1] * embeddings[-2] + embeddings[-1]

        for edge_operator in ['Average','Hadamard','Weighted-L1','Weighted-L2']:
            try:
                train_edges, train_edges_features_mtx, train_edges_labels = generate_train_set(graph_train, num_edge_sample, prev_embedding, edge_operator)
                test_edges, test_edges_features_mtx, test_edges_labels = generate_test_set(graph_test, cur_embedding, edge_operator)

                LR_clf = build_clf(train_edges_features_mtx, train_edges_labels)

                print("Edge Operator: {}".format(edge_operator))
                predict_edges_labels = pred_links(test_edges_features_mtx, LR_clf)
                precision_recall(list(predict_edges_labels), list(test_edges_labels))
                
            except:
                print("Edge Operator: {} ERROR".format(edge_operator))
        print("------------ END: {} ---------------".format(decay))
    
    print("-----------------------END EXP DECAY ---------------------")

    
    return 

def main():
    #make embeddings for reddit 
    prefix = '/z/pujat/576/data/reddit/'
    for suffix in ['reddit_1_month_dir','reddit_1_week_undir','reddit_equal_monthly_dir', 'reddit_equal_weekly_dir']:
        name = prefix + suffix + '.pkl'
        print("RUNNING: {}".format(name))
    
        with open(name, 'rb') as file:
            graphs = pickle.load(file)
        graph_train = graphs[-2]
        graph_test = graphs[-1]

        chi_list = []
        heat_print_list = []
        taus_list = []
        for e, g in enumerate(graphs[:-1]): #last embedding used for link prediction
            chi, heat_print, taus = graphwave_alg(g, np.linspace(0,200,32), taus='auto', verbose=True)
            chi_list.append(chi)
            heat_print_list.append(heat_print)
            taus_list.append(taus)
            print("Completed: {}/{}".format(e,len(graphs[:-1])))
        
        save_name = prefix + suffix + '_embeddings.npy'
        np.save(save_name, np.asarray(chi_list))
        tester(chi_list) 
        print("END RUNNING: {}".format(name))
    
main() 
