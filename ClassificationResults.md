# CLASSIFICATION RESULTS

1. BaseData + feature selection (346679, 10233)
	a. 1D CNN - NA
	b. MLP - 0.2486 (+/- 0.01)
	c. Bernouille NB - 0.04 (+/- 0.00)
	d. Adaboost - 0.14 (+/- 0.01)
	e. Random Forest - 0.21 (+/- 0.00)
	f. Extra Trees - 0.20 (+/- 0.00)
	g. Decision Trees - 0.19 (+/- 0.00)
	h. SVM (Linear) - 
	i. SVM(Kernel=rbf) - 

2. BaseData + feature selection + PCA (346679, 4600)
	a. 1D CNN - NA
	b. MLP - 0.25 (+/- 0.01)
	c. Bernouille NB - Accuracy: 0.12 (+/- 0.00)
	d. Adaboost - Accuracy: 0.12 (+/- 0.05)
	e. Random Forest - Accuracy: 0.21 (+/- 0.00)
	f. Extra Trees - Accuracy: 0.19 (+/- 0.00)
	g. Decision Trees - Accuracy: 0.17 (+/- 0.00)
	h. SVM (Linear) - 
	i. SVM(Kernel=rbf) - 


3. BaseData + feature selection + SAE (346679, 10233 -> 346679, 256)
	a. 1D CNN - NA
	b. MLP - 
	c. Bernouille NB - Accuracy: 0.14 (+/- 0.01)
	d. Adaboost - Accuracy: 0.14 (+/- 0.00)
	e. Random Forest - Accuracy: 0.14 (+/- 0.00)
	f. Extra Trees - Accuracy: 0.06 (+/- 0.00)
	g. Decision Trees - Accuracy: 0.06 (+/- 0.00)
	h. SVM (Linear) - 
	i. SVM(Kernel=rbf) - 

4. BaseData + feature selection + VAE (346679, 10233 -> 346679, 256)
	a. 1D CNN - NA
	b. MLP - 
	c. Bernouille NB - 
	d. Adaboost - 
	e. Random Forest - 
	f. Extra Trees - 
	g. Decision Trees - 
	h. SVM (Linear) - 
	i. SVM(Kernel=rbf) - 