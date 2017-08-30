## Notes about the project.

### Tsne

1. Since tsne of the entire dataset may take forever, we divide the dataset into chunks and then perform tsne on it. For this chunk of data, we run the clustering and get optimal clusters. Then the clusters are merged.
    - Have issue in merging the clusters. Need to work on how to find a criteria for cluster merging.
2. 