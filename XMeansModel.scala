import org.apache.spark.mllib.clustering.{KMeans, DistanceMeasure}


/**
 * A clustering model for K-means. Each point belongs to the cluster with the closest center.
 */
class XMeansModel (val clusterCenters: Array[Vector],
  val distanceMeasure: String) {

    def k: Int = clusterCenters.length

}
