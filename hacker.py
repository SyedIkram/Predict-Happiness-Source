#Uncomment this if you are running on jupyter notebook
import findspark
findspark.init()

import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('Emotion prediction').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.3' # make sure we have Spark 2.3+
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

#Data Cleaning
df_train = pd.read_csv('hm_train.csv')
df_train['cleaned_hm'] = df_train['cleaned_hm'].apply(lambda x: str(x).strip())
df_train['reflection_period'] = df_train['reflection_period'].apply(lambda x: int(str(x)[0:len(str(x))-1]))
df_train['cleaned_hm'] = df_train['cleaned_hm'].str.replace(r'\r\r\n','').replace(r'\r\r','')

spark_train= spark.createDataFrame(df_train)
spark_train = spark_train.filter(~isnull('hmid'))
spark_train = spark_train.filter(~isnull('reflection_period'))
spark_train = spark_train.filter(~isnull('cleaned_hm'))
spark_train = spark_train.filter(~isnull('num_sentence'))
spark_train = spark_train.filter(~isnull('predicted_category'))

#For mapping labels
prediction_scores = spark_train.groupBy("predicted_category").count().orderBy(col("count").desc())
pd_df_train =prediction_scores.toPandas()
pd_df_train['predict_score'] = np.arange(len(pd_df_train))
spark_df = spark.createDataFrame(pd_df_train)
spark_df = spark_df.drop('count')
spark_df = spark_df.selectExpr("predicted_category as predicted_category_table", "predict_score as predict_score")

#Tokenizing and Vectorizing
tok = Tokenizer(inputCol="cleaned_hm", outputCol="words")
review_tokenized = tok.transform(spark_train)

stopword_rm = StopWordsRemover(inputCol='words', outputCol='words_nsw')
review_tokenized = stopword_rm.transform(review_tokenized)

cv = CountVectorizer(inputCol='words_nsw', outputCol='tf')
cvModel = cv.fit(review_tokenized)
count_vectorized = cvModel.transform(review_tokenized)

idf_ngram = IDF().setInputCol('tf').setOutputCol('tfidf')
tfidfModel_ngram = idf_ngram.fit(count_vectorized)
tfidf_df = tfidfModel_ngram.transform(count_vectorized)

word_indexer_pc = StringIndexer(inputCol="predicted_category", outputCol="predicted_category_new", handleInvalid="error")

#Splitting the training data into training data and validation data
splits = tfidf_df.randomSplit([0.8,0.2],seed=100)
train = splits[0]
val = splits[1]

#Building the pipeline for the model
hm_assembler = VectorAssembler(inputCols=[ "tfidf"], outputCol="features")
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0,labelCol="predicted_category_new",featuresCol = "features")
hm_pipeline = Pipeline(stages=[hm_assembler, word_indexer_pc, lr])

#To get the best paramter values using CrossValidator
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.3]).addGrid(lr.elasticNetParam, [0.0, 0.1]).build()
crossval = CrossValidator(estimator=hm_pipeline,estimatorParamMaps=paramGrid,\
            evaluator=MulticlassClassificationEvaluator(labelCol="predicted_category_new", predictionCol="prediction",metricName="accuracy"),numFolds=5)

model = crossval.fit(train)
prediction_train = model.transform(val)

prediction_train_final = prediction_train.select('hmid','cleaned_hm','num_sentence','predicted_category','prediction')
prediction_train_final = prediction_train_final.withColumn("prediction", prediction_train_final["prediction"].cast(IntegerType()))
result_df_train= prediction_train_final.join(spark_df, prediction_train_final.prediction == spark_df.predict_score)
result_df_train = result_df_train.orderBy('hmid', ascending=False)

evaluator = MulticlassClassificationEvaluator(labelCol="predicted_category_new", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(prediction_train)
print("Model Accuracy = " + str(accuracy))

#Test Data
df_test = pd.read_csv('hm_test.csv')
df_test['cleaned_hm'] = df_test['cleaned_hm'].apply(lambda x: str(x).strip())
df_test['reflection_period'] = df_test['reflection_period'].apply(lambda x: int(str(x)[0:len(str(x))-1]))
df_test['cleaned_hm'] = df_test['cleaned_hm'].str.replace(r'\r\r\n','').replace(r'\r\r','')
spark_test= spark.createDataFrame(df_test)

tok_test = Tokenizer(inputCol="cleaned_hm", outputCol="words")
review_tokenized_test = tok_test.transform(spark_test)

stopword_rm_test = StopWordsRemover(inputCol='words', outputCol='words_nsw')
review_tokenized_test = stopword_rm_test.transform(review_tokenized_test)

cv_test = CountVectorizer(inputCol='words_nsw', outputCol='tf')
cvModel_test = cv_test.fit(review_tokenized)
count_vectorized_test = cvModel_test.transform(review_tokenized_test)

idf_ngram_test = IDF().setInputCol('tf').setOutputCol('tfidf')
tfidfModel_ngram_test = idf_ngram_test.fit(count_vectorized_test)
tfidf_df_test = tfidfModel_ngram_test.transform(count_vectorized_test)

test = model.transform(tfidf_df_test)
test_final = test.select('hmid','reflection_period','cleaned_hm','num_sentence','prediction')
test_final = test_final.withColumn("prediction", test_final["prediction"].cast(IntegerType()))

result_df_test= test_final.join(spark_df, test_final.prediction == spark_df.predict_score)
result_df_test = result_df_test.orderBy('hmid', ascending=True)
# result_df_test.show()
output = result_df_test.selectExpr("hmid","predicted_category_table as predicted_category")

#Saving it to the csv
output_pd = output.toPandas()

output_pd.to_csv("submission_new.csv", header=True, index=False)
