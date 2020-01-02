# XMeans
Spark-Scala Implementaion of XMeans

This implementation uses Spark-Scala's KMeans package and determain if a cluster should divided using Baysian Information Criterion. The idea for this implementation came from Dan Pelleg and Andrew Moore's [XMeans paper](https://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf). This implementation does not use the k-d tree discussed in the paper, and uses Spark's RDD to store the datapoints. 

To use this implementation simply put the `XMeans.scala` and `XMeansModel.scala` file in your working directory. The class functions exactly like [Apache Spark's KMeans class](https://spark.apache.org/docs/latest/ml-clustering.html) except there is no need to specify the number of clusters, instead you specify the maximum number of clusters you are willing to compute (Note: The number of clusters found is nearly always lower than the KMax.) An example for use would be as follows. 

```Java
val model = new XMeans().setKMax(12).run(dataset)
centers = model.clusterCenters
```
After this the centers variable will contain all the clusters that the XMeans algorithm calculated. 
