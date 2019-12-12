import numpy as np
import networkx as nx
import math
import scipy.sparse as sps
from heapq import *
from scipy.spatial.distance import euclidean
from sklearn import metrics
import matplotlib.pyplot as plt
import random
import pdb
import sys
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

MONTHLY = 0
WEEKLY = 1

AVERAGE = 0
HADAMARD = 1
L1 = 2
L2 = 3
CONCAT = 4

def combine_embeddings_sum(emb):
    '''
        Given embeddings emb, combine them by just adding them together.
    '''
    combined_embeddings = np.sum(emb, axis=0)

    return combined_embeddings

def combine_embeddings_exp_decay(emb, theta):
    '''
        Given embeddings emb, combine them with an exponential decay.
    '''
    decayed_embeddings = exp_decay(emb, theta)
    combined_embeddings = np.sum(decayed_embeddings, axis=0)

    return combined_embeddings

def exp_decay(emb, theta):
    '''
    Given a vector of size T (time),
    Multiple all entries with diminishing factors (where
    the latest entry will not be diminished)
    theta > 0
    '''
    T = len(emb)
    emb_dec = np.zeros_like(emb)
    for i in range(T):
        emb_dec[T-1 - i] = emb[T-1 - i] * math.pow(math.e, (-i * theta)) #/ 2.0

    return emb_dec

def perform_link_prediction(emb, test):
    '''
    emb: node embeddings
    test: Last snapshot graph whose edges are to be predicted.
    Returns prediction and ground truth as arrays
    '''
    # Get the num of edges in the test (T-th) graph. This number will be used
    # to generate our predicted edges.
    k = get_number_of_edges(test)

    # Get the ground truth connections
    truth = nx.to_numpy_matrix(test)

    pred_graph = get_prediction_graph(emb, k)
    # Get the corresponding numpy matrix, but make sure all node ids are included
    pred = nx.to_numpy_matrix(pred_graph, nodelist=list(range(0, len(emb))))

    # Now evaluate predictions wrt truth.
    truth_arr = np.asarray(truth.flatten()).reshape(-1)
    pred_arr = np.asarray(pred.flatten()).reshape(-1)

    return truth_arr, pred_arr

def perform_link_prediction_n2v(emb, test):
    '''
    emb: node embeddings
    test: Last snapshot graph whose edges are to be predicted.
    Returns prediction and ground truth as arrays
    '''
    # Get the num of edges in the test (T-th) graph. This number will be used
    # to generate our predicted edges.
    k = get_number_of_edges(test)

    # Get the ground truth connections
    truth = nx.to_numpy_matrix(test)
    truth1 = truth[0:-1, 0:-1]
    # truth2 = truth[1:, 1:]

    pred_graph = get_prediction_graph(emb, k)

    # Get the corresponding numpy matrix, but make sure all node ids are included
    pred = nx.to_numpy_matrix(pred_graph, nodelist=list(range(0, len(emb))))


    # Now evaluate predictions wrt truth.
    truth1_arr = np.asarray(truth1.flatten()).reshape(-1)
    # truth2_arr = np.asarray(truth2.flatten()).reshape(-1)
    pred_arr = np.asarray(pred.flatten()).reshape(-1)

    return truth1_arr, pred_arr


def evaluate(truth_arr, pred_arr):
    print(type(truth_arr))
    print(np.shape(truth_arr))
    print(np.shape(pred_arr))

    cm = metrics.confusion_matrix(truth_arr, pred_arr)
    print(cm)

    print(metrics.classification_report(truth_arr, pred_arr, digits=4))

    map = metrics.average_precision_score(truth_arr, pred_arr)
    print('Mean Average Precision: {}'.format(map))

    fpr, tpr, thresholds = metrics.roc_curve(truth_arr, pred_arr)
    roc_auc = metrics.auc(fpr, tpr)
    print('Area Under ROC Curve: {}'.format(roc_auc))

    return map, fpr, tpr, roc_auc

def get_number_of_edges(g):
    '''
    Assuming g is a networkx graph, return number of edges
    '''
    return g.number_of_edges()

# def get_prediction_graph(emb, k):
#     '''
#     First, get k closest pairs from embedding matrix.
#     Then, extract the predicted graph.
#     '''
#     print('Find k-closest edges as predictions...')
#     k_closest = find_k_closest(emb, k)
#     pred_graph = nx.Graph()
#     print('Build the graph from the k-closest...')
#     for i in range(len(k_closest)):
#         v1, v2 = k_closest[i].get_indices()
#         print('{},{}'.format(v1,v2))
#         pred_graph.add_edge(v1, v2)
#
#     return pred_graph
#
# def find_k_closest(emb, k):
#     '''
#     Finds k closest pairs of the given embedding.
#     Returns a list of custom data structure, IndexedDistance.
#     '''
#     # Create priority queue
#     prio = []
#     n = len(emb)
#
#     n = len(emb)
#     for i in range(n):
#         for j in range(i+1, n):
#             d = euclidean(emb[i], emb[j])
#             id = IndexedDistance(i, j, d)
#             heappush(prio, id)
#
#         if(len(prio) > k):
#             prio = prio[:k]
#
#     # Now prio have the smallest k point pairs with their distances.
#     return prio

def load_reality_mining_snapshots(mode : int = MONTHLY):
    '''
    Reality Mining Dataset, loaded from saved sparse matrices.
    '''
    if(mode == MONTHLY):
        T = 11      # For Reality Mining, there are 11 monthly snapshots
        path = './RM_sparse/Monthly/vc-month-'
        sparse_graphs = []
        for i in range(T):
            snp = sps.load_npz('{}{}.npz'.format(path, i))
            sparse_graphs.append(snp)

        train, test = sparse_graphs[:T-1], sparse_graphs[T-1]
        return train, test
    elif(mode == WEEKLY):
        T = 51      # For Reality Mining, there are 11 monthly snapshots
        path = './RM_sparse/Weekly/vc-week-'
        sparse_graphs = []
        for i in range(T):
            snp = sps.load_npz('{}{}.npz'.format(path, i))
            sparse_graphs.append(snp)

        train, test = sparse_graphs[:T-1], sparse_graphs[T-1]
        return train, test
    else:
        print('Invalid Processing Mode. Please use MONTHLY (0) or WEEKLY (1).')
        return None

def get_reality_mining_vc_embeddings(mode: int = 0):
    '''
    Load the pre-computed LINE embeddings
    '''
    if(mode == MONTHLY):
        T = 11
        emb_path = './RM_emb/vc_month/em-month-'
        embs = []
        for i in range(T-1):    # Read the embeddings, except the last one.
            emb = np.load('{}{}.npz'.format(emb_path, i))
            # print(list(emb.keys()))
            # print(emb['arr_0'])
            # Embeddings are in the arr0 key
            embs.append(emb['arr_0'])

    elif(mode == WEEKLY):
        T = 51
        emb_path = './RM_emb/vc_week/em-vc-week-'
        embs = []
        for i in range(T-1):    # Read the embeddings, except the last one.
            emb = np.load('{}{}.npy'.format(emb_path, i))
            # print(list(emb.keys()))
            # print(emb['arr_0'])
            # Embeddings are in the arr0 key
            # embs.append(emb['arr_0'])
            embs.append(emb)

    return np.array(embs)

def get_rm_vc_n2v_embeddings(mode: int = 0):
    if(mode == MONTHLY):
        T = 11
        emb_path = './RM_N2V_emb/vc_month_emb/vc_month_'
        embs = []
        for i in range(T-1):    # Read the embeddings, except the last one.
            emb = np.load('{}{}.npy'.format(emb_path, i))

            # if(i == 0):
            #     print(emb)
            #     print(np.shape(emb))
            #     print(emb.astype(float))
            # print(list(emb.keys()))
            # print(emb['arr_0'])
            # Embeddings are in the arr0 key
            embs.append(emb.astype(float))

    elif(mode == WEEKLY):
        T = 51
        emb_path = './RM_N2V_emb/vc_week_emb/vc_week_'
        embs = []
        for i in range(T-1):    # Read the embeddings, except the last one.
            emb = np.load('{}{}.npy'.format(emb_path, i))
            # print(list(emb.keys()))
            # print(emb['arr_0'])
            # Embeddings are in the arr0 key
            # embs.append(emb['arr_0'])
            embs.append(emb.astype(float))

    return np.array(embs)


def perform_link_prediction_linear_clf(emb, test):
    '''
    emb: node embeddings
    test: Last snapshot graph whose edges are to be predicted.
    Returns prediction and ground truth as arrays
    '''
    # Get the num of edges in the test (T-th) graph. This number will be used
    # to generate our predicted edges.
    k = get_number_of_edges(test)

    # Get the ground truth connections
    truth = nx.to_numpy_matrix(test)

    pred_graph = get_prediction_graph(emb, k)
    # Get the corresponding numpy matrix, but make sure all node ids are included
    pred = nx.to_numpy_matrix(pred_graph, nodelist=list(range(0, len(emb))))

    # Now evaluate predictions wrt truth.
    truth_arr = np.asarray(truth.flatten()).reshape(-1)
    pred_arr = np.asarray(pred.flatten()).reshape(-1)

    return truth_arr, pred_arr

def prepare_training_data_for_clf(train_graph, emb, mode):
    '''
    train_graph: networkx graph of T-1 th snapshot
    emb: embeddings for the T-1 th snapshot
    Prepare the data as follows:
        sample n edges and n nonedges.
        concat(emb_v1 , emb_v2) ; label = +1 if it's an edge
        concat(emb_v1 , emb_v2) ; label = -1 if it's not an edge
    '''
    trn = []
    labels_trn = []
    edges = list(train_graph.edges)
    nodes = list(train_graph.nodes)
    maxnode = np.max(nodes)
    num_nonedges = len(edges)

    # Add the edges
    for u,v in edges:
        if(mode == AVERAGE):
            edge_emb = (emb[u] + emb[v]) / 2.0
        elif(mode == HADAMARD):
            edge_emb = np.multiply(emb[u], emb[v])
        elif(mode == L1):
            edge_emb = np.subtract(emb[u], emb[v])
        elif(mode == L2):
            edge_emb = np.subtract(emb[u], emb[v]) * np.subtract(emb[u], emb[v])
        elif(mode == CONCAT):
            edge_emb = np.concatenate((emb[u], emb[v]), axis=None)
        # print(np.shape(edge_emb))
        trn.append(edge_emb)
        labels_trn.append(1.0)

    # Add the non-edges
    for i in range(num_nonedges):
        u = random.randint(0, maxnode)
        v = random.randint(0, maxnode)
        if(mode == AVERAGE):
            edge_emb = (emb[u] + emb[v]) / 2.0
        elif(mode == HADAMARD):
            edge_emb = np.multiply(emb[u], emb[v])
        elif(mode == L1):
            edge_emb = np.subtract(emb[u], emb[v])
        elif(mode == L2):
            edge_emb = np.subtract(emb[u], emb[v]) * np.subtract(emb[u], emb[v])
        elif(mode == CONCAT):
            edge_emb = np.concatenate((emb[u], emb[v]), axis=None)
        trn.append(edge_emb)
        labels_trn.append(0.0)

    # print(len(trn))
    # print(len(labels_trn))

    trn, labels_trn = shuffle_set(trn, labels_trn)

    return trn, labels_trn

def prepare_test_data_for_clf(test_graph, emb, mode):
    '''
    test_graph: networkx graph of T-1 th snapshot
    emb: combined embeddings up to this point
    Prepare the data as follows:
        sample n edges and n nonedges.
        concat(emb_v1 , emb_v2) ; label = +1 if it's an edge
        concat(emb_v1 , emb_v2) ; label = -1 if it's not an edge
    '''

    edges = list(test_graph.edges)
    nodes = list(test_graph.nodes)
    maxnode = np.max(nodes)
    num_nonedges = len(edges)

    tst = []
    labels_tst = []

    # Add the edges
    for u,v in edges:
        if(mode == AVERAGE):
            edge_emb = (emb[u] + emb[v]) / 2.0
        elif(mode == HADAMARD):
            edge_emb = np.multiply(emb[u], emb[v])
        elif(mode == L1):
            edge_emb = np.subtract(emb[u], emb[v])
        elif(mode == L2):
            edge_emb = np.subtract(emb[u], emb[v]) * np.subtract(emb[u], emb[v])
        elif(mode == CONCAT):
            edge_emb = np.concatenate((emb[u], emb[v]), axis=None)
        # print(np.shape(edge_emb))
        tst.append(edge_emb)
        labels_tst.append(1.0)

    # Add the non-edges
    for i in range(num_nonedges):
        u = random.randint(0, maxnode)
        v = random.randint(0, maxnode)
        if(mode == AVERAGE):
            edge_emb = (emb[u] + emb[v]) / 2.0
        elif(mode == HADAMARD):
            edge_emb = np.multiply(emb[u], emb[v])
        elif(mode == L1):
            edge_emb = np.subtract(emb[u], emb[v])
        elif(mode == L2):
            edge_emb = np.subtract(emb[u], emb[v]) * np.subtract(emb[u], emb[v])
        elif(mode == CONCAT):
            edge_emb = np.concatenate((emb[u], emb[v]), axis=None)
        tst.append(edge_emb)
        labels_tst.append(0.0)

    tst, labels_tst = shuffle_set(tst, labels_tst)

    # print(len(tst))
    # print(len(labels_tst))
    return tst, labels_tst

def shuffle_set(a, b):
    '''
    a is data and b is labels.
    shuffle both while preserving the connections.
    '''

    combined = list(zip(a, b))
    random.shuffle(combined)

    a[:], b[:] = zip(*combined)
    return a, b

if(__name__ == '__main__'):
    mode = int(sys.argv[1])
    comb_mode = int(sys.argv[2])
    print('Starting execution for mode ={}, comb_mode ={}'.format(mode, comb_mode))

    train, test = load_reality_mining_snapshots(mode=mode)

    #Get the T-1 th snapshot as train graph
    train_graph = nx.from_scipy_sparse_matrix(train[len(train)-1])

    # Get the last snapshot graph as a networkx graph.
    test_graph = nx.from_scipy_sparse_matrix(test)

    # import embeddings
    emb = get_reality_mining_vc_embeddings(mode=mode)           # LINE
    # emb = get_rm_vc_n2v_embeddings(mode=mode)                 # node2vec

    print('Embeddings loaded.')

    # Choose a combination scheme for embeddings...
    comb = None     # comb is the resulting combined embedding matrix
    comb_train = None
    if(comb_mode == 0):
        # No combination. Use the last embedding you got.
        comb = emb[len(emb)-1]
        comb_train = emb[len(emb)-1]
    elif(comb_mode == 1):
        # Sum
        comb = combine_embeddings_sum(emb)
        comb_train = combine_embeddings_sum(emb[:-1])
    elif(comb_mode == 2):
        # Exp decay
        comb = combine_embeddings_exp_decay(emb, 0.9)
        comb_train = combine_embeddings_exp_decay(emb[:-1], 0.9)
    else:
        print('Invalid combination mode. Use 0 1 2: (None, Sum, Exp)')

    # Then, use combined embeddings for temporal link prediction task
    # on the test graph, which is the T-th snapshot.
    print('Embeddings combined. Performing temporal link prediction..')

    # Prepare training data for classifier
    # last_emb = emb[len(emb)-1]


    modes = [AVERAGE, HADAMARD, L1, L2, CONCAT]

    for mode in modes:
        print('Running mode {}'.format(mode))
        trn, labels_trn = prepare_training_data_for_clf(train_graph, comb_train, mode)

        # Prepare test data for classifier
        # DEFINITELY USE combined embeddings - comb.
        tst, labels_tst = prepare_test_data_for_clf(test_graph, comb, mode)

        # clf = LinearRegression()
        clf = LogisticRegression(random_state = 0, max_iter=5000,
                                 solver='liblinear', verbose=1, tol=1e-6)

        clf.fit(trn, labels_trn)
        pred = clf.predict(tst)
        pred_binary = np.where(pred > 0.5, 1, 0)

        # print(pred_binary)

        mapscore, fpr, tpr, roc_auc = evaluate(labels_tst, pred_binary)
