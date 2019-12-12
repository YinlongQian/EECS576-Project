disp_avlbl = True
import os
if os.name == 'posix' and 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib

    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('./')
from joblib import Parallel, delayed
from .dynamic_graph_embedding import DynamicGraphEmbedding
from dynamicgem.utils import plot_util, graph_util, dataprep_util
from dynamicgem.visualization import plot_dynamic_sbm_embedding
from dynamicgem.graph_generation import dynamic_SBM_graph
from dynamicgem.evaluation import evaluate_link_prediction, evaluate_graph_reconstruction

from keras.layers import Input, Dense, Lambda, merge, Subtract 
from keras.models import Model, model_from_json
import keras.regularizers as Reg
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
from keras import callbacks
from keras import backend as KBack
from .dnn_utils import *
import tensorflow as tf
from argparse import ArgumentParser
from time import time
import operator


class DynAE(DynamicGraphEmbedding):

    def __init__(self, d, hyper_dict=None, **kwargs):
        """ Initialize the DynAE class

        Args:
            d: dimension of the embedding
            beta: penalty parameter in matrix B of 2nd order objective
            nu1: L1-reg hyperparameter
            nu2: L2-reg hyperparameter
            n_prev_graphs: lookback for previous graphs
            n_units: vector of length K-1 containing #units in hidden
                     layers of encoder/decoder, not including the units
                     in the embedding layer
            rho: bounding ratio for number of units in consecutive layers (< 1)
            n_iter: number of iterations for embedding
            xeta: sgd step size parameter
            n_batch: minibatch size for SGD or Adam
            modelfile: Files containing previous encoder and decoder models
            weightfile: Files containing previous encoder and decoder weights
            savefilesuffix: suffix for saving the files
        """
        self._d = d
        hyper_params = {
            'method_name': 'dynAE',
            'actfn': 'relu',
            'modelfile': None,
            'weightfile': None,
            'savefilesuffix': None
        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        for key in hyper_dict:
            if key == 'd':
                pass 
            self.__setattr__('_%s' % key, hyper_dict[key])

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self._d)

    def learn_embeddings(self, graphs):
        self._node_num = graphs[0].number_of_nodes()
        t1 = time()
        ###################################
        # TensorFlow wizardry
        config = tf.ConfigProto()
        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True
        # Only allow a total of half the GPU memory to be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        # Create a session with the above options specified.
        KBack.tensorflow_backend.set_session(tf.Session(config=config))
        ###################################
        # Generate encoder, decoder and autoencoder
        self._num_iter = self._n_iter
        self._encoder = get_encoder(self._node_num, self._d,
                                    self._n_units,
                                    self._nu1, self._nu2,
                                    self._actfn)
        self._decoder = get_decoder(self._node_num, self._d,
                                    self._n_units,
                                    self._nu1, self._nu2,
                                    self._actfn)
        self._autoencoder = get_autoencoder(self._encoder, self._decoder)

        # Initialize self._model
        # Input
        x_in = Input(shape=(self._node_num,), name='x_in')
        x_pred = Input(
            shape=(self._node_num,),
            name='x_pred'
        )
        # Process inputs
        [x_hat, y] = self._autoencoder(x_in)
        # Outputs
        x_diff = Subtract()([x_hat, x_in])

        # Objectives
        def weighted_mse_x(y_true, y_pred):
            """ Hack: This fn doesn't accept additional arguments.
                      We use y_true to pass them.
                y_pred: Contains x_hat - x_pred
                y_true: Contains b
            """
            return KBack.sum(
                KBack.square(y_pred * y_true[:, 0:self._node_num]),
                axis=-1
            )

        # Model
        self._model = Model(input=[x_in, x_pred], output=x_diff)
        sgd = SGD(lr=self._xeta, decay=1e-5, momentum=0.99, nesterov=True)
        adam = Adam(lr=self._xeta, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self._model.compile(optimizer=sgd, loss=weighted_mse_x)
        self._model.summary()
        
        # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        # pdb.set_trace()
        history = self._model.fit_generator(
            generator=batch_generator_dynae(
                graphs,
                self._beta,
                self._n_batch,
                self._n_prev_graphs,
                True
            ),
            nb_epoch=self._num_iter,
            samples_per_epoch=(
                                      graphs[0].number_of_nodes() * self._n_prev_graphs
                              ) // self._n_batch,
            verbose=1
            # callbacks=[tensorboard]
        )
        loss = history.history['loss']
        # Get embedding for all points
        if loss[0] == np.inf or np.isnan(loss[0]):
            print('Model diverged. Assigning random embeddings')
            self._Y = np.random.randn(self._node_num, self._d)
        else:
            self._Y, self._next_adj = model_batch_predictor_dynae(
                self._autoencoder,
                graphs[len(graphs) - self._n_prev_graphs:],
                self._n_batch
            )
        t2 = time()
        # Save the autoencoder and its weights
        if self._weightfile is not None:
            saveweights(self._encoder, self._weightfile[0])
            saveweights(self._decoder, self._weightfile[1])
        if self._modelfile is not None:
            savemodel(self._encoder, self._modelfile[0])
            savemodel(self._decoder, self._modelfile[1])
        if self._savefilesuffix is not None:
            saveweights(self._encoder,
                        '../weights/encoder_weights_' + self._savefilesuffix + '.hdf5')
            saveweights(self._decoder,
                        '../weights/decoder_weights_' + self._savefilesuffix + '.hdf5')
            savemodel(self._encoder,
                      '../weights/encoder_model_' + self._savefilesuffix + '.json')
            savemodel(self._decoder,
                      '../weights/decoder_model_' + self._savefilesuffix + '.json')
            # Save the embedding
            np.savetxt('../weights/embedding_' + self._savefilesuffix + '.txt',
                       self._Y)
            np.savetxt('../weights/next_pred_' + self._savefilesuffix + '.txt',
                       self._next_adj)
        return self._Y, (t2 - t1)

    def get_embeddings(self):
        return self._Y

    def get_edge_weight(self, i, j, embed=None, filesuffix=None):
        if embed is None:
            if filesuffix is None:
                embed = self._Y
            else:
                embed = np.loadtxt('embedding_' + filesuffix + '.txt')
        if i == j:
            return 0
        else:
            S_hat = self.get_reconst_from_embed(embed[(i, j), :], filesuffix)
            return (S_hat[i, j] + S_hat[j, i]) / 2

    def get_reconstructed_adj(self, embed=None, node_l=None, filesuffix=None):
        if embed is None:
            if filesuffix is None:
                embed = self._Y
            else:
                embed = np.loadtxt('embedding_' + filesuffix + '.txt')
        S_hat = self.get_reconst_from_embed(embed, filesuffix)
        return graphify(S_hat)

    def get_reconst_from_embed(self, embed, filesuffix=None):
        if filesuffix is None:
            return self._decoder.predict(embed, batch_size=self._n_batch)
        else:
            try:
                decoder = model_from_json(open('./intermediate/decoder_model_' + filesuffix + '.json').read())
            except:
                print('Error reading file: {0}. Cannot load previous model'.format(
                    'decoder_model_' + filesuffix + '.json'))
                exit()
            try:
                decoder.load_weights('./intermediate/decoder_weights_' + filesuffix + '.hdf5')
            except:
                print('Error reading file: {0}. Cannot load previous weights'.format(
                    'decoder_weights_' + filesuffix + '.hdf5'))
                exit()
            return decoder.predict(embed, batch_size=self._n_batch)

    def predict_next_adj(self, node_l=None):
        if node_l is not None:
            # pdb.set_trace()
            return self._next_adj[node_l]
        else:
            return self._next_adj


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

    args = parser.parse_args()
    epochs = args.epochs
    dim_emb = args.embeddimension
    lookback = args.lookback
    length = args.timelength

    if length < lookback + 5:
        length = lookback + 5
    if args.testDataType == 'sbm_rp':
        node_num = 10000
        community_num = 500
        node_change_num = 100
        dynamic_sbm_series = dynamic_SBM_graph.get_random_perturbation_series(node_num,
                                                                              community_num,
                                                                              length,
                                                                              node_change_num)
        dynamic_embedding = DynAE(
            d=100,
            beta=5,
            n_prev_graphs=lookback,
            nu1=1e-6,
            nu2=1e-6,
            n_units=[500, 300, ],
            rho=0.3,
            n_iter=1000,
            xeta=0.005,
            n_batch=500,
            modelfile=['./intermediate/enc_model.json', './intermediate/dec_model.json'],
            weightfile=['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5'],
        )
        dynamic_embedding.learn_embeddings([g[0] for g in dynamic_sbm_series])
        plt.clf()
        plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding(dynamic_embedding.get_embeddings(),
                                                              dynamic_sbm_series)
        plt.savefig('result/visualization_DynRNN_rp.png')
        plt.show()

    elif args.testDataType == 'sbm_cd':
        node_num = 1000
        community_num = 2
        node_change_num = args.nodemigration
        dynamic_sbm_series = dynamic_SBM_graph.get_community_diminish_series_v2(node_num,
                                                                                community_num,
                                                                                length,
                                                                                1,  # communitiy to dimisnish
                                                                                node_change_num
                                                                                )
        dynamic_embedding = DynAE(
            d=dim_emb,
            beta=5,
            n_prev_graphs=lookback,
            nu1=1e-6,
            nu2=1e-6,
            n_units=[500, 300, ],
            rho=0.3,
            n_iter=epochs,
            xeta=args.learningrate,
            n_batch=args.batch,
            modelfile=['./intermediate/enc_model.json', './intermediate/dec_model.json'],
            weightfile=['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5'],
            savefilesuffix="testing"
        )
        graphs = [g[0] for g in dynamic_sbm_series]
        embs = []

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/dynAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            result = Parallel(n_jobs=4)(delayed(dynamic_embedding.learn_embeddings)(graphs[:temp_var]) for temp_var in
                                        range(lookback + 1, length + 1))
            for i in range(len(result)):
                embs.append(np.asarray(result[i][0]))

            for temp_var in range(lookback + 1, length + 1):
                emb, _ = dynamic_embedding.learn_embeddings(graphs[:temp_var])
                embs.append(emb)

            plt.figure()
            plt.clf()
            plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding_v2(embs[-5:-1], dynamic_sbm_series[-5:])
            plt.savefig('./' + outdir + '/V_DynAE_nm' + str(args.nodemigration) + '_l' + str(length) + '_epoch' + str(
                epochs) + '_emb' + str(dim_emb) + '.pdf', bbox_inches='tight', dpi=600)
            plt.show()

        if args.hypertest == 1:
            fname = 'epoch' + str(args.epochs) + '_bs' + str(args.batch) + '_lb' + str(args.lookback) + '_eta' + str(
                args.learningrate) + '_emb' + str(args.embeddimension)
        else:
            fname = 'nm' + str(args.nodemigration) + '_l' + str(length) + '_emb' + str(dim_emb)

        if args.exp == 'lp':
            evaluate_link_prediction.expLP(
                graphs,
                dynamic_embedding,
                1,
                outdir + '/',
                fname,
            )

    elif args.testDataType == 'academic':
        print("datatype:", args.testDataType)

        dynamic_embedding = DynAE(
            d=dim_emb,
            beta=5,
            n_prev_graphs=lookback,
            nu1=1e-6,
            nu2=1e-6,
            n_units=[500, 300, ],
            rho=0.3,
            n_iter=epochs,
            xeta=1e-5,
            n_batch=100,
            modelfile=['./intermediate/enc_modelacdm.json', './intermediate/dec_modelacdm.json'],
            weightfile=['./intermediate/enc_weightsacdm.hdf5', './intermediate/dec_weightsacdm.hdf5'],
            savefilesuffix="testingacdm"
        )

        sample = args.samples
        if not os.path.exists('./test_data/academic/pickle'):
            os.mkdir('./test_data/academic/pickle')
            graphs, length = dataprep_util.get_graph_academic('./test_data/academic/adjlist')
            for i in range(length):
                nx.write_gpickle(graphs[i], './test_data/academic/pickle/' + str(i))
        else:
            length = len(os.listdir('./test_data/academic/pickle'))
            graphs = []
            for i in range(length):
                graphs.append(nx.read_gpickle('./test_data/academic/pickle/' + str(i)))

        G_cen = nx.degree_centrality(graphs[29])  # graph 29 in academia has highest number of edges
        G_cen = sorted(G_cen.items(), key=operator.itemgetter(1), reverse=True)
        node_l = []
        i = 0
        while i < sample:
            node_l.append(G_cen[i][0])
            i += 1
        for i in range(length):
            graphs[i] = graph_util.sample_graph_nodes(graphs[i], node_l)

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/dynAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            evaluate_link_prediction.expLP(graphs[-args.timelength:],
                                           dynamic_embedding,
                                           1,
                                           outdir + '/',
                                           'lb_' + str(lookback) + '_l' + str(args.timelength) + '_emb' + str(
                                               dim_emb) + '_samples' + str(sample),
                                           n_sample_nodes=sample
                                           )

    elif args.testDataType == 'hep':
        print("datatype:", args.testDataType)
        dynamic_embedding = DynAE(
            d=dim_emb,
            beta=5,
            n_prev_graphs=lookback,
            nu1=1e-6,
            nu2=1e-6,
            n_units=[500, 300, ],
            rho=0.3,
            n_iter=epochs,
            xeta=1e-8,
            n_batch=int(args.samples / 10),
            modelfile=['./intermediate/enc_modelhep.json', './intermediate/dec_modelhep.json'],
            weightfile=['./intermediate/enc_weightshep.hdf5', './intermediate/dec_weightshep.hdf5'],
            savefilesuffix="testinghep"
        )

        if not os.path.exists('./test_data/hep/pickle'):
            os.mkdir('./test_data/hep/pickle')
            files = [file for file in os.listdir('./test_data/hep/hep-th') if '.gpickle' in file]
            length = len(files)
            graphs = []
            for i in range(length):
                G = nx.read_gpickle('./test_data/hep/hep-th/month_' + str(i + 1) + '_graph.gpickle')

                graphs.append(G)

            #get maximum number of nodes
            total_nodes = graphs[-1].number_of_nodes()

            #make sure all nodes are present in all graphs
            for i in range(length):
                for j in range(total_nodes):
                    if j not in graphs[i].nodes():
                        graphs[i].add_node(j)

            #resave the graphs with normalized node counts
            for i in range(length):
                nx.write_gpickle(graphs[i], './test_data/hep/pickle/' + str(i))
        else:
            length = len(os.listdir('./test_data/hep/pickle'))
            graphs = []
            for i in range(length):
                graphs.append(nx.read_gpickle('./test_data/hep/pickle/' + str(i)))

        #given the degree centrality
        #we are going to do sampling. 
        sample = args.samples
        G_cen = nx.degree_centrality(graphs[-1])  # graph 29 in academia has highest number of edges
        G_cen = sorted(G_cen.items(), key=operator.itemgetter(1), reverse=True)
        node_l = []
        i = 0
        while i < sample:
            node_l.append(G_cen[i][0])
            i += 1
        for i in range(length):
            graphs[i] = graph_util.sample_graph_nodes(graphs[i], node_l)

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/dynAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            evaluate_link_prediction.expLP(graphs[-args.timelength:],
                                           dynamic_embedding,
                                           1,
                                           outdir + '/',
                                           'lb_' + str(lookback) + '_l' + str(args.timelength) + '_emb' + str(
                                               dim_emb) + '_samples' + str(sample),
                                           n_sample_nodes=sample
                                           )

    elif args.testDataType == 'AS':
        print("datatype:", args.testDataType)
        dynamic_embedding = DynAE(
            d=dim_emb,
            beta=5,
            n_prev_graphs=lookback,
            nu1=1e-6,
            nu2=1e-6,
            n_units=[500, 300, ],
            rho=0.3,
            n_iter=epochs,
            xeta=1e-5,
            n_batch=int(args.samples / 10),
            modelfile=['./intermediate/enc_modelAS.json', './intermediate/dec_modelAS.json'],
            weightfile=['./intermediate/enc_weightsAS.hdf5', './intermediate/dec_weightsAS.hdf5'],
            savefilesuffix="testingAS"
        )

        files = [file for file in os.listdir('./test_data/AS/as-733') if '.gpickle' in file]
        length = len(files)
        graphs = []

        for i in range(length):
            G = nx.read_gpickle('./test_data/AS/as-733/month_' + str(i + 1) + '_graph.gpickle')
            graphs.append(G)

        sample = args.samples
        G_cen = nx.degree_centrality(graphs[-1])  # graph 29 in academia has highest number of edges
        G_cen = sorted(G_cen.items(), key=operator.itemgetter(1), reverse=True)
        node_l = []
        i = 0
        while i < sample:
            node_l.append(G_cen[i][0])
            i += 1
        for i in range(length):
            graphs[i] = graph_util.sample_graph_nodes(graphs[i], node_l)

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/dynAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            evaluate_link_prediction.expLP(graphs[-args.timelength:],
                                           dynamic_embedding,
                                           1,
                                           outdir + '/',
                                           'lb_' + str(lookback) + '_l' + str(args.timelength) + '_emb' + str(
                                               dim_emb) + '_samples' + str(sample),
                                           n_sample_nodes=sample
                                           )

    elif args.testDataType == 'enron':
        print("datatype:", args.testDataType)

        dynamic_embedding = DynAE(
            d=dim_emb,
            beta=5,
            n_prev_graphs=lookback,
            nu1=1e-6,
            nu2=1e-6,
            n_units=[500, 300, ],
            rho=0.3,
            n_iter=epochs,
            xeta=1e-8,
            n_batch=20,
            modelfile=['./intermediate/enc_modelenron.json', './intermediate/dec_modelenron.json'],
            weightfile=['./intermediate/enc_weightsenron.hdf5', './intermediate/dec_weightsenron.hdf5'],
            savefilesuffix="testingAS"
        )

        files = [file for file in os.listdir('./test_data/enron') if 'week' in file]
        length = len(files)
        graphs = []

        for i in range(length):
            G = nx.read_gpickle('./test_data/enron/week_' + str(i) + '_graph.gpickle')
            graphs.append(G)

        sample = graphs[0].number_of_nodes()
        print(sample)

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/dynAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            evaluate_link_prediction.expLP(graphs[-args.timelength:],
                                           dynamic_embedding,
                                           1,
                                           outdir + '/',
                                           'lb_' + str(lookback) + '_l' + str(args.timelength) + '_emb' + str(
                                               dim_emb) + '_samples' + str(sample),
                                           n_sample_nodes=sample
                                           )
