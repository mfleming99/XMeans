package org.mf

import util.Random

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector}
import org.apache.spark.rdd.RDD

class XMeans (
  private var kMax: Int,
  private var maxIterations: Int,
  private var initializationMode: String,
  private var initializationSteps: Int,
  private var epsilon: Double,
  private var seed: Long,
  private var distanceMeasure: String) extends Serializable {

  /**
     * Constructs a XMeans instance with default parameters: {kMax: 12, maxIterations: 20,
     * initializationMode: "k-means||", initializationSteps: 2, epsilon: 1e-4, seed: random,
     * distanceMeasure: "euclidean"}.
     */
  def this() = this(12, 20, KMeans.K_MEANS_PARALLEL, 2, 1e-4, Random.nextLong, "euclidean")


  /**
   * Maximum number of clusters that XMeans can make.
   */
  def getKMax: Int = kMax

  /**
   * Set maximum number of clusters that XMeans can make.
   */
  def setKMax(kMax: Int): this.type = {
    require(kMax > 0,
      s"Number of clusters must be positive but got ${kMax}")
    this.kMax = kMax
    this
  }

  /**
   * Maximum number of iterations allowed.
   */
  def getMaxIterations: Int = maxIterations

  /**
   * Set maximum number of iterations allowed. Default: 20.
   */
  def setMaxIterations(maxIterations: Int): this.type = {
    require(maxIterations >= 0,
      s"Maximum of iterations must be nonnegative but got ${maxIterations}")
    this.maxIterations = maxIterations
    this
  }

  /**
   * The initialization algorithm. This can be either "random" or "k-means||".
   */
  def getInitializationMode: String = initializationMode

  /**
   * Set the initialization algorithm. This can be either "random" to choose random points as
   * initial cluster centers, or "k-means||" to use a parallel variant of k-means++
   * (Bahmani et al., Scalable K-Means++, VLDB 2012). Default: k-means||.
   */
  def setInitializationMode(initializationMode: String): this.type = {
    require(initializationMode == "random" || initializationMode == "k-means||",
      s"Initialization mode must be 'random' or 'k-means', but got '${initializationMode}'")
    this.initializationMode = initializationMode
    this
  }

  /**
   * Number of steps for the k-means|| initialization mode
   */
  def getInitializationSteps: Int = initializationSteps

  /**
   * Set the number of steps for the k-means|| initialization mode. This is an advanced
   * setting -- the default of 2 is almost always enough. Default: 2.
   */
  def setInitializationSteps(initializationSteps: Int): this.type = {
    require(initializationSteps > 0,
      s"Number of initialization steps must be positive but got ${initializationSteps}")
    this.initializationSteps = initializationSteps
    this
  }


  /**
   * The distance threshold within which we've consider centers to have converged.
   */
  def getEpsilon: Double = epsilon

  /**
   * Set the distance threshold within which we've consider centers to have converged.
   * If all centers move less than this Euclidean distance, we stop iterating one run.
   */
  def setEpsilon(epsilon: Double): this.type = {
    require(epsilon >= 0,
      s"Distance threshold must be nonnegative but got ${epsilon}")
    this.epsilon = epsilon
    this
  }

  /**
   * The random seed for cluster initialization.
   */
  def getSeed: Long = seed

  /**
   * Set the random seed for cluster initialization.
   */
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  /**
   * The distance suite used by the algorithm.
   */
  def getDistanceMeasure: String = distanceMeasure

/**
 * Set the distance suite used by the algorithm.
 */
def setDistanceMeasure(distanceMeasure: String): this.type = {
  //TODO Add some sort of verification that the distanceMeasure is valid
  //DistanceMeasure.validateDistanceMeasure(distanceMeasure)
  this.distanceMeasure = distanceMeasure
  this
}


  def run(data: RDD[Vector]): Array[org.apache.spark.mllib.linalg.Vector] = {
    var model = new KMeans()
                  .setK(1)
                  .setMaxIterations(maxIterations)
                  .setInitializationMode(initializationMode)
                  .setInitializationSteps(initializationSteps)
                  .setEpsilon(epsilon)
                  .setDistanceMeasure(distanceMeasure)
                  .run(data)
    var centers = model.clusterCenters
    var oldCenterCount = 0.toLong
    while (oldCenterCount != centers.length && centers.length < kMax / 2) {
      oldCenterCount = centers.length
      centers = centers.flatMap(c => centroidSplitter(c, data, model))
      model = new KMeans()
        .setK(centers.length)
        .setMaxIterations(0)
        .setInitializationMode(initializationMode)
        .setEpsilon(epsilon)
        .setInitialModel(new KMeansModel(centers))
        .setSeed(seed)
        .setDistanceMeasure(distanceMeasure)
        .run(data)
      centers = model.clusterCenters
    }
    centers
  }

  private def calculateBICModel(
    data: RDD[Vector],
    model: KMeansModel): Double = {
      val bic = model.clusterCenters.map(center => calculateBICCluster(center, data, model, model.k)).sum
      bic
  }

  private def calculateBICCluster(
    center: Vector,
    data: RDD[Vector],
    model: KMeansModel,
    k: Int): Double = {
      val d = center.size
      val r = data.count()
      val rn = model.predict(data)
        .filter(_ == model.clusterCenters.indexOf(center))
        .count()
      val l1 = -1 * rn * Math.log(2 * Math.PI) / 2
      val l2 = -1 * rn * d * Math.log(varianceMLE(data, model, k)) / 2
      val l3 = -1 * (rn - k) / 2
      val l4 = rn * Math.log(rn)
      val l5 = -1 * rn * Math.log(r)
      val l6 = -1 * Math.log(r) * (k - 1 + (k * d) + varianceMLE(data, model, k))
      val bic = l1 + l2 + l3 + l4 + l5 + l6
    bic
  }

  private def centroidSplitter(
    center: Vector,
    data: RDD[Vector],
    model: KMeansModel): Array[Vector] = {
      val clusterData = model.predict(data)
        .zip(data)
        .filter(_._1 == model.clusterCenters.indexOf(center))
        .map(x => x._2)
        .cache()
      val modelk1 = new KMeans()
        .setK(1)
        .setMaxIterations(maxIterations)
        .setInitializationMode(initializationMode)
        .setInitializationSteps(initializationSteps)
        .setEpsilon(epsilon)
        .setSeed(seed)
        .setDistanceMeasure(distanceMeasure)
        .run(clusterData)
      val modelk2 = new KMeans()
        .setK(2)
        .setMaxIterations(maxIterations)
        .setInitializationMode(initializationMode)
        .setInitializationSteps(initializationSteps)
        .setEpsilon(epsilon)
        .setSeed(seed)
        .setDistanceMeasure(distanceMeasure)
        .run(clusterData)
      if (modelk2.k == 2) {
        if (calculateBICModel(clusterData, modelk1) > calculateBICModel(clusterData, modelk2)) {
          modelk1.clusterCenters
        } else {
          modelk2.clusterCenters
        }
      } else {
        modelk1.clusterCenters
      }
    }

  private def varianceMLE(
    data: RDD[Vector],
    model: KMeansModel,
    k: Long): Double = {
    val variance = model.computeCost(data) / (data.count() - k).toDouble
    variance
  }
}