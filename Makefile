sample:
	ipython nbconvert --to=python ./haha/Untitled2.ipynb
	python Untitled2.py

LINE-Embeddings:
	ipython nbconvert --to=python ./embeddings/LINE_Embeddings.ipynb
	python LINE_Embeddings.py

College-Graphs:
	ipython nbconvert --to=python ./graphs/CollegeMsg_Graphs.ipynb
	python CollegeMsg_Graphs.py

LINE-Amazon-Link-Prediction:
	ipython nbconvert --to=python ./link-prediction/LINE_link_prediction_Amazon-Food.ipynb
	python LINE_link_prediction_Amazon-Food.py

LINE-Email-Link-Prediction:
	ipython nbconvert --to=python ./link-prediction/LINE_link_prediction_Email-EU.ipynb
	python LINE_link_prediction_Email-EU.py

LINE-College-Link-Prediction:
	ipython nbconvert --to=python ./link-prediction/LINE_link_prediction_College-Msg.ipynb
	python LINE_link_prediction_College-Msg.py

DeepWalk-Amazon-Link-Prediction:
	ipython nbconvert --to=python ./link-prediction/DW_link_prediction_Amazon-Food.ipynb
	python DW_link_prediction_Amazon-Food.py

DeepWalk-RM-Link-Prediction:
	ipython nbconvert --to=python ./link-prediction/deepwalk_linear_classifier_link_prediction_RM.ipynb
	python deepwalk_linear_classifier_link_prediction_RM.py

N2V-Amazon-Link-Prediction:
	ipython nbconvert --to=python ./link-prediction/N2V_link_prediction_Amazon-Food.ipynb
	python N2V_link_prediction_Amazon-Food.py

GraphWave-College-Link-Prediction:
	ipython nbconvert --to=python ./link-prediction/CollegeMsg_Test.ipynb
	python CollegeMsg_Test.py


