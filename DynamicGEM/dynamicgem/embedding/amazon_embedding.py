disp_avlbl = True
import os
if os.name == 'posix' and 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib

    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import pickle
sys.path.append('./')
from joblib import Parallel, delayed
from dynamicgem.embedding import dynamic_graph_embedding as DynamicGraphEmbedding
from dynamicgem.utils import plot_util, graph_util, dataprep_util
from dynamicgem.visualization import plot_dynamic_sbm_embedding
from dynamicgem.graph_generation import dynamic_SBM_graph
from dynamicgem.evaluation import evaluate_link_prediction, evaluate_graph_reconstruction
from dynamicgem.embedding.dynAE import *
from dynamicgem.embedding.dynAERNN import * 
from dynamicgem.embedding.dynRNN import *

from keras.layers import Input, Dense, Lambda, merge, Subtract 
from keras.models import Model, model_from_json
import keras.regularizers as Reg
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
from keras import callbacks
from keras import backend as KBack
from dynamicgem.embedding.dnn_utils import *
import tensorflow as tv 
from argparse import ArgumentParser
from time import time
import operator
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M%S', level=logging.INFO)

def setup_dir(outdir,testDataType,method):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outdir = outdir + '/' + testDataType
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    outdir = outdir + '/' + method
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    return outdir

if __name__ == '__main__':
    parser = ArgumentParser(description='Learns node embeddings for a sequence of graph snapshots')
    parser.add_argument('-t', '--testDataType',
                        default='sbm_cd',
                        type=str,
                        help='Type of data to test the code')
    parser.add_argument('-c', '--criteria',
                        default='degree',
                        type=str,
                        help='Node Migration criteria')
    parser.add_argument('-rc', '--criteria_r',
                        default=1,
                        type=int,
                        help='Take highest centrality measure to perform node migration')
    parser.add_argument('-l', '--timelength',
                        default=10,
                        type=int,
                        help='Number of time series graph to generate')
    parser.add_argument('-lb', '--lookback',
                        default=2,
                        type=int,
                        help='number of lookbacks')
    parser.add_argument('-eta', '--learningrate',
                        default=1e-4,
                        type=float,
                        help='learning rate')
    parser.add_argument('-bs', '--batch',
                        default=100,
                        type=int,
                        help='batch size')
    parser.add_argument('-nm', '--nodemigration',
                        default=10,
                        type=int,
                        help='number of nodes to migrate')
    parser.add_argument('-iter', '--epochs',
                        default=250,
                        type=int,
                        help='number of epochs')
    parser.add_argument('-emb', '--embeddimension',
                        default=128,
                        type=int,
                        help='embedding dimension')
    parser.add_argument('-rd', '--resultdir',
                        type=str,
                        default='./results_link_all',
                        help="result directory name")
    parser.add_argument('-sm', '--samples',
                        default=2000,
                        type=int,
                        help='samples for test data')
    parser.add_argument('-ht', '--hypertest',
                        default=0,
                        type=int,
                        help='hyper test')
    parser.add_argument('-exp', '--exp',
                        default='lp',
                        type=str,
                        help='experiments (lp, emb)')
    parser.add_argument('-method','--method',
                        default='dynAE',
                        type=str,
                        help='which method (dynAE, dynRNN, dynAERNN')


    args = parser.parse_args()
    epochs = args.epochs
    dim_emb = args.embeddimension
    lookback = args.lookback
    length = args.timelength
    meth = args.method
    sample = args.samples
   
    if length < lookback + 5:
        length = lookback + 5

    if args.testDataType == 'amazon_agg':
        
        with open('/z/pujat/576/data/amazon/amazon_agg_weighted_2011_2014.pkl','rb') as file:
            graphs = pickle.load(file)

        save_suffix = "amazon_agg"
    elif args.testDataType == "dblp" or args.testDataType == "DBLP":

        file_path_1 = '/z/pujat/576/data/dblp/dblp_directed_aggregated.pkl' 
        with open(file_path_1,'rb') as file:
            graphs = pickle.load(file)
        save_suffix = "dblp_agg"
        print("dblp file path: ",file_path_1)    

    elif args.testDataType in ("dblp_1year", "DBLP_1year"):
        print("1 year")

    G_cen = nx.degree_centrality(graphs[-1])  
    G_cen = sorted(G_cen.items(), key=operator.itemgetter(1), reverse=True)
    node_l = []
    i = 0
    while i < sample:
        node_l.append(G_cen[i][0])
        i += 1
    for i in range(length):
        graphs[i] = graph_util.sample_graph_nodes(graphs[i], node_l)

    outdir = setup_dir(args.resultdir,args.testDataType,args.method)
        
    logging.info("Dataset %s  -- Method %s",args.testDataType, args.method)
    logging.info("Lookback %s", lookback) 
    if args.method == 'dynAE' or args.method == 'dynae':
        
        hyper_dict = {
            "beta":5,
            "n_prev_graphs":lookback,
            "nu1":1e-6,
            "nu2":1e-6,
            "n_units":[500,300],
            "rho":0.3,
            "n_iter":epochs,
            "xeta":1e-5,
            "n_batch":100,
            "modelfile":['./intermediate/enc_model_'+ save_suffix +'.json',
                       './intermediate/dec_model_'+ save_suffix + '.json'],
            "weightfile":['./intermediate/enc_weights_'+ save_suffix + '.hdf5',
                        './intermediate/dec_weights_' + save_suffix + '.hdf5'],
            "savefilesuffix":save_suffix}
        dynamic_embedding = DynAE(
            d=dim_emb,hyper_dict=hyper_dict
            ) 


    if args.method == 'dynAERNN' or args.method == 'DynAERNN':

        dynamic_embedding = DynAERNN(
            d=dim_emb,
            beta=5,
            n_prev_graphs=lookback,
            nu1=1e-6,
            nu2=1e-6,
            n_aeunits=[500, 300],
            n_lstmunits=[500, dim_emb],
            rho=0.3,
            n_iter=epochs,
            xeta=1e-3,
            n_batch=100,
            modelfile=['./intermediate/enc_modelAERNN_' + save_suffix + '.json',
                       './intermediate/dec_modelAERNN_'+ save_suffix + '.json'],
            weightfile=['./intermediate/enc_weightsAERNN_' + save_suffix + '.hdf5',
                        './intermediate/dec_weightsAERNN_' + save_suffix + '.hdf5'],
            savefilesuffix=save_suffix
        )

    if args.method in ('dynRNN', 'DynRNN', 'dynrnn'):
         
        dynamic_embedding = DynRNN(
            d=dim_emb,  # 128,
            beta=5,
            n_prev_graphs=lookback,
            nu1=1e-6,
            nu2=1e-6,
            n_enc_units=[500, 300],
            n_dec_units=[500, 300],
            rho=0.3,
            n_iter=epochs,
            xeta=1e-3,
            n_batch=int(args.samples / 10),
            modelfile=['./intermediate/enc_modelRNN_' + save_suffix +'.json',
                       './intermediate/dec_modelRNN_' + save_suffix + '.json'],
            weightfile=['./intermediate/enc_weightsRNN_' + save_suffix + '.hdf5', 
                        './intermediate/dec_weightsRNN_' + save_suffix + '.hdf5'],
            savefilesuffix=save_suffix
        )

    
    evaluate_link_prediction.expLP(graphs[-args.timelength:],
                                   dynamic_embedding,
                                   1,
                                   outdir + '/',
                                   'lb_' + str(lookback) + '_l' + str(args.timelength) + '_emb' + str(
                                       dim_emb) + '_samples' + str(sample),
                                   n_sample_nodes=sample
                                   ) 

