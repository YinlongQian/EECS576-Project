'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import scipy.sparse as sps
import pickle
import sys

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

input_file = sys.argv[1]
output_dir = sys.argv[2]

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	with open(input_file, 'rb') as file:
		graphs = pickle.load(file)
	# only one time operation

	return graphs

def learn_embeddings(nx_G, walks, output_file):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	result = []
	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.wv.save_word2vec_format(output_file)
	return

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	graph_list = read_graph()
	cnt = 0 
	for nx_G in graph_list:
		print(nx_G.size())
		G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
		G.preprocess_transition_probs()
		walks = G.simulate_walks(args.num_walks, args.walk_length)
		output_file = output_dir+'output_equal_month_' + str(cnt) + '.emb'
		learn_embeddings(nx_G, walks, output_file)
		cnt += 1

if __name__ == "__main__":
	args = parse_args()
	main(args)
