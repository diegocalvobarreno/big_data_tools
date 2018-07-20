%spark2.spark

val rdd = sc.parallelize( Array(1, 2, 3, 4, 5) ) // Definir RDD 
val df = rdd.toDF() //Transformar a df
