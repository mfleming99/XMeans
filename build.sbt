scalaVersion := "2.12.10"

name := "XMeans"
organization := "org.mflem.XMeans"
version := "1.1"

githubOwner := "mfleming99"
githubRepository := "XMeans"

libraryDependencies += "org.apache.spark" %% "spark-core" %"2.4.5"
libraryDependencies += "org.apache.spark" %% "spark-mllib" %"2.4.5"
libraryDependencies += "org.scalactic" %% "scalactic" % "3.2.0"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.0" % "test"
libraryDependencies += "com.holdenkarau" %% "spark-testing-base" % "2.4.5_0.14.0" % "test"