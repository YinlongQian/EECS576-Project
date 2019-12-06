SRC= ./src/main_pickle.py
ALL= $(SRC) 

deepwalk: 
	python ./src/main_pickle.py --p 1 --q 1 --directed --input graphs/msg_1_month_dir.pkl --output emb_deepwalk/

node2vec:
	python ./src/main_pickle.py --p 4 --q 1 --directed --input graphs/msg_1_month_dir.pkl --output emb_node2vec/

all.zip: $(ALL)
	zip all.zip $(ALL)

clean:
	\rm -rf all.zip
