%spark2.spark

// Ejemplo simple realizado para extrapolar

import org.apache.spark.ml.linalg.Vectors
val df = spark.createDataFrame(Seq(
    (0, 60),
    (0, 56),
    (0, 54),
    (0, 62),
    (0, 61),
    (0, 53),
    (0, 55),
    (0, 62),
    (0, 64),
    (1, 73),
    (1, 78),
    (1, 67),
    (1, 68),
    (1, 78)
)).toDF("defecto" , "temperatura")

 
//Definir el modelo mediante tuberias

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}

// Definir características
val features = new VectorAssembler()
  .setInputCols(Array("temperatura"))
  .setOutputCol("features")

// Definir modelo a utilizar
val lr = new LinearRegression().setLabelCol("defecto")

// Crear una tuberia que asocie el modelo con la secuencia de tratamiento de datos
val pipeline = new Pipeline().setStages(Array(features, lr))

//Ejecutar el modelo
val model = pipeline.fit(df)

 
//Mostrar resultados del modelo

val linRegModel = model.stages(1).asInstanceOf[LinearRegressionModel]

println(s"RMSE:  ${linRegModel.summary.rootMeanSquaredError}")
println(s"r2:    ${linRegModel.summary.r2}")
println(s"Model: Y = ${linRegModel.coefficients(0)} * X + ${linRegModel.intercept}")



//Mostrar prediciones

val result = model.transform(data).select("temperatura", "defecto", "prediction")
result.show()
