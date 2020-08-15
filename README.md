# Predict-Happiness-Source
Happiness Source Predictor coding challenge by HackerEarth

# Problem
SmileDB is a corpus of more than 100,000 happy moments crowd-sourced via Amazon’s Mechanical Turk. Each worker is given the following task: What made you happy today? Reflect on the past 24 hours, and recall three actual events that happened to you that made you happy. Write down your happy moment in a complete sentence. (Write three such moments.)

The goal of the corpus is to advance the understanding of the causes of happiness through text-based reflection.
Based on the happy moment statement you have to predict the category of happiness, i.e. the source of happiness which is typically either of the following: 'bonding', 'achievement', 'affection', 'leisure', 'enjoy_the_moment', 'nature', 'exercise'.

# Data description
The training set contains more than 60,000 samples, while your trained model will be tested on more than 40,000 samples.

| Column Name | Column Description | Column Datatype |
| --- | --- | --- |
| Hmid | Id of the person | Int64 |
| Reflection_period | The time of happiness | Object |
| Cleaned_hm | Happiness Statement Made | Object |
| Num_sentence | No. of sentences present in the person’s statement. | Int64 |
| Predicted_category | Source of happiness | object |

# Solution
Technical Stack Requirements: Spark 2.3+, Python 3.5+, Numpy, Pandas
Command to run the file: # spark-submit hacker.py

Built a supervised learning classification model i.e Logistic Regression using spark mllib to predict the cause of happiness through text-based reflection and performed an evaluation using MulticlassClassificationEvaluator. The model achieved an accuracy of ~82%.
