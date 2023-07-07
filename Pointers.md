## Notes about the project.

### General

1. There are failed analyses. These need to be run once entire framework is ready. When producing results, we can run everything once and meanwhile write paper.  
2. The model's for **LLE** and **TSNE** will not be saved in the inital run. Once the accuracies are determined from the results, we will rerun with the optimal configuration and then store the model.
3. Clustering accuracy needs to be estimated based on NMI, ARI and the ACC parameters. 

### Lineage Analysis
1. There are 2 encoding methods
	- Old => Each of the sequence is encoded as 3 bit vector.
	- new => No encoding is performed.
2. Each executable is identified by the md5 value. Whether the executable is analyzed or not is determined by checking the presence of family_name.fasta. 
	- In case the file is absent, for all members of the family which are analyzed, we generate the MSA file i.e fasta file.
	- In case the file is present, it means the current executable is freshly encountered and analysis is done. Then a file family_name.fasta_add is generated and supplied to MAFFT software with --add argument.


### Tsne

1. Since tsne of the entire dataset may take forever, we divide the dataset into chunks and then perform tsne on it. For this chunk of data, we run the clustering and get optimal clusters. Then the clusters are merged.
    - Have issue in merging the clusters. Need to work on how to find a criteria for cluster merging.
2. 


### LLE

#### Results format

Results is a dictionary with the key being the index for chunk of the matrix.
The value is a list with 2 values 
 
 1. first value corresponds to the **DBSCAN** accuracies which is a list.
    - The list elements which are the accuracies of the model for each value of n_neighbors in lle.
    - Each element in the above list corresponds to the accuracies of dbscan when run with varying eps and min_samples values.
 2. second value corresponds to the **HDBSCAN** accuracies.
    - The list elements which are the accuracies of the model for each value of n_neighbors in lle.
    - Each element in the above list is a tuple where the first value is cluster accuracy and the second value is the input labels.
    - The cluster accuracy is a dict and contains the accuracies of each cluster.
    - The input labels are the corresponding **avclass** labels of the malware. 

## Time Taken
Test malware on Lab machine : 22