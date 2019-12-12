disp_avlbl = True
import os
if os.name == 'posix' and 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib

    matplotlib.use('Agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib.pyplot as plt
import sys
import pickle
sys.path.append('./')
from joblib import Parallel, delayed

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from dynamicgem.embedding import dynamic_graph_embedding as DynamicGraphEmbedding
from dynamicgem.utils import plot_util, graph_util, dataprep_util
from dynamicgem.visualization import plot_dynamic_sbm_embedding, plot_dynamic_embedding
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
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from argparse import ArgumentParser
from time import time
import operator
import logging



logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M%S', level=logging.INFO)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    parser.add_argument('-l', '--timelength', default=10,
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
                        help='experiments (lp, emb,save_emb)')
    parser.add_argument('-method','--method',
                        default='dynAE',
                        type=str,
                        help='which method (dynAE, dynRNN, dynAERNN')
    
    parser.add_argument('-is_agg','--is_aggregated',
                        default="false",
                        type=str,
                        help='use an aggregated model or snapshot')

    parser.add_argument('-t_int', '--time_interval',
                        default='week',
                        type =str,
                        help='snapshot spacing: day or week')

    parser.add_argument('-is_dir', '--is_directed',
                        default="true",
                        type=str,
                        help='Use the directed or undirected version of the nx graphs')

    args = parser.parse_args()
    epochs = args.epochs
    dim_emb = args.embeddimension
    lookback = args.lookback
    length = args.timelength
    meth = args.method
    sample = args.samples
    time_interval = args.time_interval

    logging.info("Time interval %s",time_interval)
    is_agg = str2bool(args.is_aggregated)
    is_dir = str2bool(args.is_directed)
    if length < lookback + 5:
        length = lookback + 5


    if time_interval == "month":
        with open('/z/pujat/576/data/email_eu/email_1_month_dir.pkl', 'rb') as file:
            graphs = pickle.load(file)
        save_suffix = "email_eu_Monthly"
    
    elif time_interval == 'week': 
        with open('/z/pujat/576/data/email_eu/email_1_week_dir.pkl','rb') as file:
            graphs = pickle.load(file)

        save_suffix = "email_eu_Weekly"
    
    elif time_interval == 'equal_monthly':

        with open('/z/pujat/576/data/email_eu/email_equal_monthly_dir.pkl','rb') as file:
            graphs = pickle.load(file)

        save_suffix = "email_equal_monthly"

    elif time_interval == 'equal_weekly':

        with open('/z/pujat/576/data/email_eu/email_equal_weekly_dir.pkl','rb') as file:
            graphs = pickle.load(file)

        save_suffix = "email_equal_weekly"
    
    else:
        logging.info("SPECIFY a valid time ") 

    logging.info("Dataset %s  -- Method %s",args.testDataType, args.method)
    logging.info("Lookback %s", lookback) 
    logging.info("time interval %s -- is_agg %s -- is_dir %s",time_interval,
                 is_agg, is_dir)
    


    #NOTE: For CollegeMsg , the last graph contains edges for 
    # years that haven't been completed;
    # this means that there are substantially less edges 
    # than the preceding time step
    # It is unfair to measure accuracy on this set b/c 
    # the ground truth is only partially available!
    logging.info("Len graph seq: %s",len(graphs))
    logging.info("Specified Length: %s", length)
    G_cen = nx.degree_centrality(graphs[-1])  
    G_cen = sorted(G_cen.items(), key=operator.itemgetter(1), reverse=True)
    node_l = []
    i = 0
    
    #PT: We want the embeddings of all nodes
    #So, I need to sample all of them. 
    
    if sample < graphs[-1].number_of_nodes():
        samples = graphs[-1].number_of_nodes()
    
    logging.info("Number of nodes (samples): %s", sample)
    while i < sample:
        try:
            node_l.append(G_cen[i][0])
            i += 1
        except:
            break 
    for i in range(length):
        graphs[i] = graph_util.sample_graph_nodes(graphs[i], node_l)

    logging.info('Resulting length:%s ', len(graphs))

    outdir = setup_dir(args.resultdir,args.testDataType,args.method)
        
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
            n_lstmunits=[300, dim_emb],
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

    if args.exp == 'emb':
        embs = []
        t1 = time()
        #Going to length + 1 ensures that the last embedding is created
        for temp_var in range(lookback + 1, length + 1):
            emb, _ = dynamic_embedding.learn_embeddings(graphs[:temp_var])
            embs.append(emb)
        
        print(dynamic_embedding._method_name + ':\n\tTraining time: %f' %(time()
                                                                          -t1))

        try:
            plt.figure()
            plt.clf()
            plot_dynamic_embedding.plot_dynamic_embedding(embs[-5:-1], graphs[-5:],t_steps=[0,1,2,3])
            im_name = save_suffix + "_" + str(lookback) + ".png" 
            plt.savefig(im_name)
        except:
            print("Couldn't save the embedding image")


    elif args.exp == 'save_emb':
        embs = []
        t1 = time()
        logging.info("look back %d length %d", lookback, length)
        
        #Don't want the embedding to train on the last graph 
        #for temp_var in range(lookback + 1, length + 1):
        for temp_var in range(lookback + 1, length ):
            logging.info("Step: %d", temp_var)
            emb, _ = dynamic_embedding.learn_embeddings(graphs[:temp_var])
            embs.append(emb)
        
        get_emb = dynamic_embedding.get_embeddings()
        next_pred = dynamic_embedding.predict_next_adj()
        print(dynamic_embedding._method_name + ':\n\tTraining time: %f' %(time()
                                                                          -t1))
        
        
        param_names =  'lb_' + str(lookback) + '_l' + str(args.timelength) + '_emb' + str(dim_emb) + '_samples' + str(sample)
        save_name = outdir + '/' + save_suffix +"_" + param_names + '.npy'
        np.save(save_name, embs)
        
        save_pred = outdir + '/' + save_suffix +"_" + param_names + '_next_pred.npy'
        np.save(save_pred, next_pred)

        save_name_2 = outdir + '/' + save_suffix + "_" + param_names +  '_gottten_embs.npy'
        print(save_name)
        print(save_pred)
        print(save_name_2)
        print(len(embs))
        np.save(save_name, embs)
        np.save(save_name_2, get_emb)

        
        try:
            plt.figure()
            plt.clf()
            plot_dynamic_embedding.plot_dynamic_embedding(embs[-5:-1], graphs[-5:],t_steps=[0,1,2,3])
            im_name = save_suffix + "_" + str(lookback) + ".png" 
            plt.savefig(im_name)
        except:
            print("Couldn't save the embedding image")

    else:
        print(evaluate_link_prediction.expLP(graphs[-args.timelength:],
                                       dynamic_embedding,
                                       1,
                                       outdir + '/',
                                       'lb_' + str(lookback) + '_l' + str(args.timelength) + '_emb' + str(
                                           dim_emb) + '_samples' + str(sample),
                                       n_sample_nodes=sample
                                       )) 
