package org.mf.XMeans

import org.scalatest.flatspec.AnyFlatSpec
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import com.holdenkarau.spark.testing.SharedSparkContext

class XMeansSpec extends AnyFlatSpec with SharedSparkContext {

  behavior of "XMeans"

  it should "have correct default parameters" in {
    val xmeans = new XMeans()
    assert(xmeans.getKMax == 12)
    assert(xmeans.getMaxIterations == 20)
    assert(xmeans.getInitializationMode == "k-means||")
    assert(xmeans.getInitializationSteps == 2)
    assert(xmeans.getEpsilon == 1e-4)
    assert(xmeans.getDistanceMeasure == "euclidean")
  }

  it should "accept setters" in {
    val xmeans = new XMeans()
      .setKMax(6)
      .setMaxIterations(10)
      .setInitializationMode("random")
      .setInitializationSteps(1)
      .setEpsilon(1e-5)
      .setDistanceMeasure("cosine")
    assert(xmeans.getKMax == 6)
    assert(xmeans.getMaxIterations == 10)
    assert(xmeans.getInitializationMode == "random")
    assert(xmeans.getInitializationSteps == 1)
    assert(xmeans.getEpsilon == 1e-5)
    assert(xmeans.getDistanceMeasure == "cosine")
  }

  it should "find three centroids" in {

    val data = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0, 6.0),
      Vectors.dense(1.0, 3.0, 0.0),
      Vectors.dense(1.0, 4.0, 6.0)
    )).cache()

    val centroids = new XMeans().run(data)
    assert(centroids.length == 3)
  }

  it should "not find more centroids than kMax" in {

    val data = sc.parallelize(
      (0 to 10000)
        .map(v => Vectors.dense(v.toDouble, v.toDouble))
    ).cache()

    val centroids = new XMeans()
      .setSeed(42)
      .run(data)
    assert(centroids.length <= 12)
  }

  it should "find approximately correct centroids" in {

    val r = new scala.util.Random(42)

    val targetCentroids = Set((10L, 10L), (10L, -10L), (-10L, 10L))

    val data = sc.parallelize(
      targetCentroids
        .flatMap(pt => (0 to 100)
          .map(_ => Vectors.dense(r.nextGaussian() + pt._1, r.nextGaussian() + pt._2))
        )
        .toSeq
    ).cache()

    val centroids = new XMeans()
      .setSeed(42)
      .run(data)
    val roundedCentroids = centroids
      .map(v => v.toArray)
      .map(v => (Math.round(v(0)), Math.round(v(1))))
      .toSet
    assert(centroids.length == 3)
    assert(targetCentroids.equals(roundedCentroids))
  }
}

