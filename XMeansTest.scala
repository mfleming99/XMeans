

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel, DistanceMeasure}
import org.apache.spark.mllib.linalg.Vectors
import java.io._
import java.nio._
import sys.process._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import java.lang.Math
import org.apache.spark.util.Utils


import scala.collection.mutable.ArrayBuffer

import org.apache.spark.annotation.Since
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.random.XORShiftRandom



class XMeans private (
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
  def this() = this(12, 20, KMeans.K_MEANS_PARALLEL, 2, 1e-4, Utils.random.nextLong(),
    DistanceMeasure.EUCLIDEAN)


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
    DistanceMeasure.validateDistanceMeasure(distanceMeasure)
    this.distanceMeasure = distanceMeasure
    this
  }

  def run(data: RDD[Vector]): Array[org.apache.spark.mllib.linalg.Vector] = {
    var model = KMeans.train(data, 2, maxIterations)
    var centers = model.clusterCenters
    var currentModelScore = calculateBICModel(data, model)
    var oldCenterCount = 0.toLong
    while (oldCenterCount != centers.length && centers.length < kMax) {
      oldCenterCount = centers.length
      centers = centers.flatMap(c => centroidSplitter(c, data, model))
      model = new KMeans()
        .setK(centers.length)
        .setInitialModel(new KMeansModel(centers))
        .run(data)
      centers = model.clusterCenters
    }
    centers
  }

  def calculateBICModel(
    data: RDD[Vector],
    model: KMeansModel): Double = {
      val bic = model.clusterCenters.map(center => calculateBICCluster(center, data, model, model.k)).sum
      bic
  }

  def calculateBICCluster(
    center: Vector,
    data: RDD[Vector],
    model: KMeansModel,
    k: Int): Double = {
      val points = model.predict(data)
        .filter(_ == model.clusterCenters.indexOf(center))
        .count()
      val bic = (-1 * points * Math.log(2 * Math.PI) / 2) - (points * center.size * Math.log(varianceMLE(data, model, k)) / 2) - (points - k / 2) + (points * Math.log(points)) - (points * Math.log(data.count())) - (((k - 1) + (center.size * k) + 1) * Math.log(data.count()) / 2)
    bic
  }

  def centroidSplitter(
    center: Vector,
    data: RDD[Vector],
    model: KMeansModel): Array[Vector] = {
      val clusterData = model.predict(data)
        .zip(data)
        .filter(_._1 == model.clusterCenters.indexOf(center))
        .map(x => x._2).cache()
      val modelk1 = KMeans.train(clusterData, 1, maxIterations)
      val modelk2 = KMeans.train(clusterData, 2, maxIterations)
      if (modelk2.k == 2) {
        if (calculateBICCluster(modelk1.clusterCenters(0), clusterData, modelk1, 1) > (calculateBICCluster(modelk2.clusterCenters(0), clusterData, modelk2, 1) + calculateBICCluster(modelk2.clusterCenters(1), clusterData, modelk2, 1))) {
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

case class DataPopulator(val dimensions: Int, val partitions: Int, val dataSetLable: Int) extends Serializable {
  val dims: Int = dimensions
  val parts: Int = partitions
  val lable: Int = dataSetLable

  def populateData(): Array[org.apache.spark.mllib.linalg.Vector] = {
    val cmd = Seq("./points", dims.toString, parts.toString, lable.toString, pointsPerPartition.toString, range.toString)
    (cmd).!(ProcessLogger(line => ()))
    val bis = new BufferedInputStream(new FileInputStream("syntheticData_" + lable + ".bin"))
    var data = byteArrToDoubleArr(Stream.continually(bis.read)
      .takeWhile(-1 !=).map(_.toByte).toArray)
      .sliding(dimensions, dimensions)
      .toArray
    var vectoredData = new Array[org.apache.spark.mllib.linalg.Vector](data.length)
    for (i <- 0 to data.length - 1) {
      vectoredData(i) = Vectors.dense(data(i))
    }
    vectoredData
  }

  def byteArrToDoubleArr(arr: Array[Byte]) : Array[Double] = {
    val times = 8
    val newArr = Array.ofDim[Double](arr.length / times)
    for (i <- 0 to newArr.length - 1) {
      newArr(i) = ByteBuffer.wrap(arr, times*i, times).order(ByteOrder.LITTLE_ENDIAN).getDouble()
    }
    newArr
  }
}

val dimensions = 2
val partitions = 4
val range = 10
val pointsPerPartition = 512
var dataPopArray = new Array[DataPopulator](partitions)

for (i <- 0 to partitions - 1) {
  dataPopArray(i) = new DataPopulator(dimensions, partitions, i)
}


val dataset = sc.parallelize(dataPopArray, partitions)
  .map(x =>x.populateData())
  .flatMap(x => x)
  .cache()

println("Result = " + new XMeans().setKMax(3).run(dataset).mkString(",\n\t"))

System.exit(0)
