# XMeans
Spark-Scala Implementaion of XMeans

This is a clustering library that tries to guess how many centroids there are, instead of using a set number like many classical clustering algorithms.


This is my attempt at implementing Dan Pelleg and Andrew Moore's [XMeans paper](https://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf). This implementation does not use the k-d tree discussed in the paper, and uses Spark's RDD to store the datapoints. 

## Install 

This package uses Scala 2.12 and Spark 2.4.5. To add this package to your sbt project, add the following two lines in your `build.sbt` file. 

```sbt
externalResolvers += "XMeans package" at "https://maven.pkg.github.com/mfleming99/XMeans"
libraryDependencies += "org.mf" %% "XMeans" % "1.2"
```

## Use
The class functions similarly to [Apache Spark's KMeans class](https://spark.apache.org/docs/latest/ml-clustering.html) except there is no need to specify the number of clusters, instead you specify the maximum number of centroids you are willing to compute (Note: The number of centroids found is nearly always lower than the KMax). An example for use would be as follows. 

```Java
val centroids = new XMeans().setKMax(12).run(dataset)
```
Now `centroids` will contain all the centriods that XMeans computed
