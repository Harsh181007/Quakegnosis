import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, DecisionTreeRegressor, GBTRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

# Create the Spark session with the MongoDB Spark Connector
spark = SparkSession\
    .builder\
    .master('local[4]')\
    .appName('quakes_etl')\
    .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:2.4.1')\
    .getOrCreate()

# Load the dataset
df_load = spark.read.csv(r"database.csv", header=True)

# Remove all fields we don't need
lst_dropped_columns = ['Depth Error', 'Time', 'Depth Seismic Stations', 'Magnitude Error', 'Magnitude Seismic Stations', 'Azimuthal Gap', 'Horizontal Distance', 
                       'Horizontal Error', 'Root Mean Square', 'Source', 'Location Source', 'Magnitude Source', 'Status']

df_load = df_load.drop(*lst_dropped_columns)

# Create a year field and add it to the df_load dataframe
df_load = df_load.withColumn('Year', year(to_timestamp('Date', 'dd/MM/yyyy')))

# Create the quakes frequency dataframe from the year and count values
df_quake_freq = df_load.groupBy('Year').count().withColumnRenamed('count', 'Counts')

# Cast string fields to double types
df_load = df_load.withColumn('Latitude', df_load['Latitude'].cast(DoubleType())) \
    .withColumn('Longitude', df_load['Longitude'].cast(DoubleType())) \
    .withColumn('Depth', df_load['Depth'].cast(DoubleType())) \
    .withColumn('Magnitude', df_load['Magnitude'].cast(DoubleType()))

# Create avg and max magnitude fields and add to df_quake_freq
df_max = df_load.groupBy('Year').max('Magnitude').withColumnRenamed('max(Magnitude)', 'Max_Magnitude')
df_avg = df_load.groupBy('Year').avg('Magnitude').withColumnRenamed('avg(Magnitude)', 'Avg_Magnitude')

# Join the max and avg DataFrames to df_quake_freq
df_quake_freq = df_quake_freq.join(df_avg, ['Year']).join(df_max, ['Year'])

# Remove records with null values
df_load = df_load.dropna()
df_quake_freq = df_quake_freq.dropna()

# Load df_load into MongoDB (Quake database)
df_load.write.format('com.mongodb.spark.sql.DefaultSource') \
    .mode('overwrite') \
    .option('uri', 'mongodb://127.0.0.1:27017/Quake.quakes') \
    .save()

# Load df_quake_freq into MongoDB (Quake database)
df_quake_freq.write.format('com.mongodb.spark.sql.DefaultSource') \
    .mode('overwrite') \
    .option('uri', 'mongodb://127.0.0.1:27017/Quake.quake_freq') \
    .save()

# Print dataframe heads
print("df_quake_freq DataFrame:")
df_quake_freq.show(5)
print("df_load DataFrame:")
df_load.show(5)

df_test = spark.read.csv(r"query.csv", header=True)
df_test.take(1)

# loading query.csv into mongodb
df_train = spark.read.format('mongo')\
    .option('spark.mongodb.input.uri', 'mongodb://127.0.0.1:27017/Quake.quakes').load()

print("df_train DataFrame:")
df_train.show(5)
print('INFO: Job ran successfully')

# eliminating unnecessary fields
df_test_clean = df_test ['time', 'latitude', 'longitude', 'mag', 'depth']
print("df_test_clean DataFrame:")
df_test_clean.show(5)

df_test_clean = df_test_clean.withColumnRenamed('time','Date',)\
    .withColumnRenamed('latitude', 'Latitude')\
    .withColumnRenamed('longitude', 'Longitude')\
    .withColumnRenamed('mag', 'Magnitude')\
    .withColumnRenamed('depth', 'Depth')
print("Revised df_test_clean DataFrame:")
df_test_clean.show(5)

df_test_clean.printSchema()

 
df_test_clean = df_test_clean.withColumn('Latitude', df_test_clean['Latitude'].cast(DoubleType()))\
    .withColumn('Longitude', df_test_clean['Longitude'].cast(DoubleType()))\
    .withColumn('Depth', df_test_clean['Depth'].cast(DoubleType()))\
    .withColumn('Magnitude', df_test_clean['Magnitude'].cast(DoubleType()))

df_test_clean.printSchema()

# creating testing and training dataframes
df_testing = df_test_clean ['Latitude', 'Longitude', 'Magnitude', 'Depth']
df_training = df_train ['Latitude', 'Longitude', 'Magnitude', 'Depth']

# preview training data
print("df_train Dataframe:")
df_training.show(5)

#preview testing data
print("df_testing Dataframe")
df_testing.show(5)

# dropping nulls
df_testing = df_testing.dropna()
df_training = df_training.dropna()


# ML model start..
# Seleting features to pass into our model and create feature vector
def evaluate_model(model, df_training, df_testing):
    pipeline = Pipeline(stages=[assembler, model])
    model_fitted = pipeline.fit(df_training)
    pred_results = model_fitted.transform(df_testing)
    rmse = evaluator.evaluate(pred_results)
    return rmse

# Prepare evaluator and assembler
assembler = VectorAssembler(inputCols=['Latitude', 'Longitude', 'Depth'], outputCol='features')
evaluator = RegressionEvaluator(labelCol='Magnitude', predictionCol='prediction', metricName='rmse')

# RandomForestRegressor
rf_reg = RandomForestRegressor(featuresCol='features', labelCol='Magnitude')
rmse_rf = evaluate_model(rf_reg, df_training, df_testing)
print(f"RandomForest RMSE: {rmse_rf}")

# LinearRegression
lr_reg = LinearRegression(featuresCol='features', labelCol='Magnitude')
rmse_lr = evaluate_model(lr_reg, df_training, df_testing)
print(f"Linear Regression RMSE: {rmse_lr}")

# DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(featuresCol='features', labelCol='Magnitude')
rmse_dt = evaluate_model(dt_reg, df_training, df_testing)
print(f"Decision Tree RMSE: {rmse_dt}")

# Gradient-Boosted Trees Regressor
gbt_reg = GBTRegressor(featuresCol='features', labelCol='Magnitude')
rmse_gbt = evaluate_model(gbt_reg, df_training, df_testing)
print(f"Gradient-Boosted Trees RMSE: {rmse_gbt}")

# Compare RMSE results
print(f"RMSE Comparison:\n RandomForest: {rmse_rf}\n Linear Regression: {rmse_lr}\n Decision Tree: {rmse_dt}\n Gradient-Boosted Trees: {rmse_gbt}")

# Choose the best model based on lowest RMSE
models_rmse = {'RandomForest': rmse_rf, 'LinearRegression': rmse_lr, 'DecisionTree': rmse_dt, 'GBT': rmse_gbt}
best_model = min(models_rmse, key=models_rmse.get)
print(f"The best model is {best_model} with RMSE: {models_rmse[best_model]}")