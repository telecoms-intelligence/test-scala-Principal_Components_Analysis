import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().appName("PCA_Example").getOrCreate()

// Use Spark to read in the Cancer_Data file.
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Cancer_Data")

// Print the Schema of the data
data.printSchema()

// Import PCA, VectorAssembler and StandardScaler from ml.feature
import org.apache.spark.ml.feature.{PCA,StandardScaler,VectorAssembler}

// Import Vectors from ml.linalg
import org.apache.spark.ml.linalg.Vectors

val colnames = (Array("mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
"mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
"radius error", "texture error", "perimeter error", "area error", "smoothness error", "compactness error",
"concavity error", "concave points error", "symmetry error", "fractal dimension error", "worst radius",
"worst texture", "worst perimeter", "worst area", "worst smoothness", "worst compactness", "worst concavity",
"worst concave points", "worst symmetry", "worst fractal dimension"))

val assembler = new VectorAssembler().setInputCols(colnames).setOutputCol("features")

// Use the assembler to transform our DataFrame to a single column: features
val output = assembler.transform(data).select($"features")

val scaler = (new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(false))

// Compute summary statistics by fitting the StandardScaler.
// Basically create a new object called scalerModel by using scaler.fit()
// on the output of the VectorAssembler
val scalerModel = scaler.fit(output)

// Normalize each feature to have unit standard deviation.
// Use transform() off of this scalerModel object to create your scaledData
val scaledData = scalerModel.transform(output)

// Now its time to use PCA to reduce the features to some principal components

// Create a new PCA() object that will take in the scaledFeatures
// and output the pcs features, use 4 principal components
// Then fit this to the scaledData
val pca = (new PCA()
  .setInputCol("scaledFeatures")
  .setOutputCol("pcaFeatures")
  .setK(4)
  .fit(scaledData))

// Once your pca has been created and fit, transform the scaledData
// Call this new dataframe pcaDF
val pcaDF = pca.transform(scaledData)

// Show the new pcaFeatures
val result = pcaDF.select("pcaFeatures")
result.show()
