# MachineLearning

Code to generate the dataset from cuckoo reports of analyzed malware binaries. 
This branch is a python3 implementation and consists of major changes due to porting. 

## Updates:
1. Clustering module is no longer necessary and hence is removed in its entirely.
2. MDS from Linear Dimensionality is not used and hence removed.
3. Standardized the usage of Utility functions and hence, its unnecessary to implement a custom logger module.
4. Non-Linear Dimensionality module is removed as only Autoencoder is being used. The code needs to be written separately based on the jupyter notebook of the same.
5. Old Code is irrelevant and hence removed. 
6. DataGeneratorKeras.py from PrepareData module is removed.
7. Many changes have been performed such as 
    a. Optimizing code.
    b. Re-writing for python 3 compatibility.
    c. Dead-code removal.

### Authors:
Satwik
