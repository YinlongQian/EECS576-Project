1. enviroment:
Anaconda3 - Python 3.7

2. requirments:
scipy==1.3.1
numpy==1.17.2
networkx==1.11
gensim==3.8.1


3. files and functions:
    1) ./src/main_pickle.py - generate node embeddings to ./emb_ fold from pickle file.
    2) ./src/main.py - generate node embeddings to ./emb_ fold from .npz file.
    3) ./src/node2vec.py - from the author of node2vec paper, the main implementation of node2vec algorithm.
    4) emb2npy.py - from emb to npy (format conversion)
    5) run_emb2npy.sh - iterate the number of snapshots to run emb2npy.py file
    6) run_main.sh - .sh file to run ./src/main.py 

4. how to run:
   
    1) first is to generate the embedding file from pickle graphs in graphs folder to emb folder:
	example: python ./src/main_pickle.py --input input_file --output output_dir --p 1 --q 1 --directed --weighted (p,q is the parameters for node2vec, --directed means the graph is directed, --weighted means the graph is weighted.
    2) run run_emb2npy.sh this file will run emb2npy.py to do format conversion.
	example: sh run_emb2npy.sh
    3) Node2Vec_linear_classfier_link_prediction.ipynb logistic classifier for link prediction.

5. make file

make deepwalk command will run the deepwalk method to generate embeddings
make node2vec command will run the node2vec method to generate embeddings
