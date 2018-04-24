
Instruction to run the project:
----------------------------------
1. Compile main.py. 
   Output: 
   a) Doc2Vec model : model_DBOW.doc2vec (PV-DBOW model) and model_DM.doc2vec (PM-DV model)
   b) score.csv : Diff similairty scores between student and refrence answers
2. Feed socres.csv to weka and classify using logistic regression with 10 fold cross valdiation.
   On using each of those scores individually and all of them together should output different prediciotn accuracy.
   Such result on my run are inside prediciton_result folder

Project Detail:
----------------------------------
Project name: dtAssesment
Input: 
	DT-Grade_v1.0_dataset.xml inside DT-Gradev1.0_data folder
 	DT-Grade_v1.0_dataset.xml  is file containg collection of quetions, answer given by student and set of refrence asnwers provided by expert. You can detail abut the data set in the paper DT-Grade_dataset pdf (Banjade, et al., 2016) inside DT-Gradev1.0_data folder

output:
a) 2 Doc2vec model inside modles folder
   	model_DBOW.doc2vec is doc2vec model using PV-DBOW approach
   	model_DM.doc2vec is doc2vec model using PV-DM approach

   	For detail about PV-DBOW and PV-DBOW apprach, read paper doc2vec-mikolov.pdf inside papers folder


b) silairty scores between student answer and the refrences answers
   socres.csv insdie ouptut foler contains different cosine simialirty scores between answer and refrences answers.
   
   DM_max_cosSim : max cosine simialrity scores between a answer and set of refrences answers using PV-DM vectors 
   DM_max_cosSim : averagge cosine simialrity scores of scores between a answer and set of refrences answers using PV-DM vectors 
   DBOW_max_cosSim : max cosine simialrity scores between a answer and set of refrences answers using PV-DBOW vectors 
   DBOW_mean_cosSim : averagge cosine simialrity scores of scores between a answer and set of refrences answers using PV-DBOW vectors 
   DM_DBOW_max_cosSim : max cosine simialrity scores between a answer and set of refrences answers using concatination of PV-DBOW  and PV-DM vectors 
   DM_DBOW_mean_cosSim : averagge cosine simialrity scores of scores between a answer and set of refrences answers using concatination of PV-DBOW  and PV-DM vectors 
   Annotation : gold label for student answers which can be correct, correct_but_incomplete, contradictory and incorrect

c) Predictor output : output of logistic regression computed using  weka inside predcition_result folder

   The result of predictor using logsitc classifer with default setting and 10 cross fold valiation looks likes as folows:

	Features: DM_max_cosSim
	Gold Label: Annotation
	Accuracy: 41.5556


	Features: DM_mean_cosSim
	Gold Label: Annotation
	Accuracy: 43.2222

	Features: DBOW_max_cosSim
	Gold Label: Annotation
	Accuracy: 40.8889

	Features: DBOW_mean_cosSim
	Gold Label: Annotation
	Accuracy: 40.8889


	Features: DM_DBOW_max_cosSim
	Gold Label: Annotation
	Accuracy: 40.8889

	Features: DM_DBOW_mean_cosSim
	Gold Label: Annotation
	Accuracy: 40.8889


	Features: DM_max_cosSim, DM_mean_cosSim, DBOW_max_cosSim, DBOW_mean_cosSim, DM_DBOW_max_cosSim, DM_DBOW_mean_cosSim
	Gold Label: Annotation 
	Accuracy: 42.1111






