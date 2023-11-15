#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">ALS-Powered Movie Recommender: Sparking Personalized Cinema Selections</h1>
# 
# ## Sai Sanwariya Narayan
# 
# This project focuses on developing a sophisticated movie recommendation system utilizing the Alternating Least Squares (ALS) algorithm within the Apache Spark framework:
# 
# 1. **Implementing ALS for Movie Recommendations**: based on user reviews, leveraging the MLlib library in Spark.
# 
# 2. **Data Management and Analysis**: Skills in handling data through splitting it into training, validation, and testing sets. This process is crucial for building robust machine learning models.
# 
# 3. **Error Calculation and Hyperparameter Tuning**: Techniques to calculate training, validation, and testing errors, and understanding the importance of tuning hyperparameters for optimal model performance.
# 
# 4. **Efficiency in Processing**: Utilizing Spark's RDD transformations and CheckPoint features to enhance processing efficiency.
# 

# ## You will need to `pip install pandas` in the terminal for this within the your environment

# In[1]:


import pyspark
import pandas as pd
import numpy as np
import math


# In[2]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType
from pyspark.sql.functions import col, column
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql import Row
from pyspark.mllib.recommendation import ALS


# In[3]:


ss=SparkSession.builder.master("local").appName("Lab6 ALS-based Recommendation Systems").getOrCreate()


# In[4]:


ss.sparkContext.setCheckpointDir("~/scratch")


# In[5]:


rating_schema = StructType([ StructField("UserID", IntegerType(), False ),                             StructField("MovieID", IntegerType(), True),                             StructField("Rating", FloatType(), True ),                             StructField("RatingID", IntegerType(), True ),                            ])


# In[6]:


ratings_DF = ss.read.csv("/storage/home/ratings_2.csv", schema=rating_schema, header=True, inferSchema=False)


# In[7]:


#ratings_DF.printSchema()


# In[8]:


ratings2_DF = ratings_DF.select("UserID","MovieID","Rating")


# In[9]:


#ratings2_DF.first()


# In[10]:


ratings2_RDD = ratings2_DF.rdd
#ratings2_RDD.take(3)


# # Spliting Data into Three Sets: Training Data, Evaluation Data, and Testing Data

# In[11]:


training_RDD, validation_RDD, test_RDD = ratings2_RDD.randomSplit([3,1,1], 19)


# ## Prepare input (UserID, MovieID) for training, validation and for testing data

# In[12]:


training_input_RDD = training_RDD.map(lambda x: (x[0], x[1]) )
validation_input_RDD = validation_RDD.map(lambda x: (x[0], x[1]) ) 
testing_input_RDD = test_RDD.map(lambda x: (x[0], x[1]) )


# # A Movie Recommendation Model 
# ## using ALS (from `PySpark.MLlib.recommendation` module) and training data. Choose a rank between 3 and 6, a randon number for the seed, 30 iterations, 0.1 regularization parameter.

# In[13]:


model = ALS.train(training_RDD, 4, seed=17, iterations=30, lambda_=0.1)


# ## Compute Training Error of the ALS recommendation model

# In[14]:


training_prediction_RDD = model.predictAll(training_input_RDD)


# In[15]:


#training_prediction_RDD.take(4)


# # Three Ways to Access Elements of a 'Row' object in an RDD. 
# We are going to demonstrate these three methods for accessing/transforming the format of the RDD so that ``(<user>, <movie>)`` is in the key position so that we can join the RDD containing the actual rating with the RDD containing the predicted rating (for calculating prediction errors).

# In[16]:


#training_RDD.take(3)


# ## Method 1: Access elements of a row using column name of the DataFrame (from which the RDD came from) using the syntax ``<row variable>[ <ColumnName> ]``

# In[17]:


training_target_output_RDD = training_RDD.map(lambda x: ( (x['UserID'], x['MovieID']), x['Rating'] ) )


# In[18]:


#training_target_output_RDD.take(3)


# ## Method 2: Access elements of a row using column name (that does not contain space) of the DataFrame schema (from which the RDD came from) using the syntax ``<row variable>.<ColumnName>"

# In[19]:


training_target_output2_RDD = training_RDD.map(lambda x: ( ( x.UserID, x.MovieID ), x.Rating ) )


# In[20]:


#training_target_output2_RDD.take(3)


# ## Method 3: Access elements of a row using column name of the DataFrame (from which the RDD came from) using the syntax ``<row variable>[<index>]`` where ``<index>`` is the integer that indicates the position of the element in the row (starting with 0 for the first element).

# In[21]:


training_target_output3_RDD = training_RDD.map(lambda x: ( (x[0], x[1]), x[2] ) )


# In[22]:


#training_target_output3_RDD.take(3)


# ## Transforming the model output of training data into the format of `( (<UserID> <MovieID>), <predictedRating> )` so that we can later join it with training target outpt RDD for computing Root Mean Square Error of predictions.

# In[23]:


training_prediction2_RDD = training_prediction_RDD.map(lambda x: ( (x[0], x[1]), x[2] ) )


# In[24]:


#training_prediction2_RDD.take(3)


# In[25]:


training_evaluation_RDD = training_target_output_RDD.join(training_prediction2_RDD)


# In[26]:


#training_evaluation_RDD.take(3)


# In[27]:


training_error = math.sqrt(training_evaluation_RDD.map(lambda z: (z[1][0] - z[1][1])**2).mean())


# In[28]:


#print(training_error)


# ## Compute Validation Errors

# In[29]:


validation_prediction_RDD = model.predictAll(validation_input_RDD).map(lambda x: ( (x[0], x[1]), x[2] ) )


# In[30]:


#validation_prediction_RDD.take(3)


# ## Joining `validation_RDD` (after transforming it into the same key value pair format, and `validation_prediction_RDD` to prepare for RMS error calculation.

# In[31]:


validation_evaluation_RDD = validation_RDD.map(lambda y: ((y[0], y[1]), y[2] ) ).join(validation_prediction_RDD)


# In[32]:


#validation_evaluation_RDD.take(3)


# ## Calculating RMS error for validation data.

# In[33]:


validation_error = math.sqrt(validation_evaluation_RDD.map(lambda z: (z[1][0] - z[1][1])**2 ).mean())


# In[34]:


#print(validation_error)


# # Hyperparameter Tuning
# ## Iterating through all possible combination of a set of values for three hyperparameters for ALS Recommendation Model:
# - rank (k)
# - regularization
# - iterations 
# ## Each hyperparameter value combination is used to construct an ALS recommendation model using training data, but evaluate using Evaluation Data
# ## The evaluation results are saved in a Pandas DataFrame 
# ``
# hyperparams_eval_df
# ``
# ## The best hyperprameter value combination is stored in 4 variables
# ``
# best_k, best_regularization, best_iterations, and lowest_validation_error
# ``

# ## Setting of hyperparameters (rank k, regularization parameter, and number of iterations) to create and evaluate ALS recommendation models to find the best model among all those created.

# In[35]:


## Initialize a Pandas DataFrame to store evaluation results of all combination of hyper-parameter settings
hyperparams_eval_df = pd.DataFrame( columns = ['k', 'regularization', 'iterations', 'validation RMS', 'testing RMS'] )
# initialize index to the hyperparam_eval_df to 0
index =0 
# initialize lowest_error
lowest_validation_error = float('inf')
# Set up the possible hyperparameter values to be evaluated
iterations_list = [15, 30]
regularization_list = [0.1, 0.2, 0.3]
rank_list = [4, 7, 10, 13]
for k in rank_list:
    for regularization in regularization_list:
        for iterations in iterations_list:
            seed = 37
            # Construct a recommendation model using a set of hyper-parameter values and training data
            model = ALS.train(training_RDD, k, seed=seed, iterations=iterations, lambda_=regularization)
            # Evaluate the model using evalution data
            # map the output into ( (userID, movieID), rating ) so that we can join with actual evaluation data
            # using (userID, movieID) as keys.
            validation_prediction_RDD= model.predictAll(validation_input_RDD).map(lambda x: ( (x[0], x[1]), x[2])   )
            validation_evaluation_RDD = validation_RDD.map(lambda y: ( ( y[0], y[1]), y[2] ) ).join(validation_prediction_RDD)
            # Calculate RMS error between the actual rating and predicted rating for (userID, movieID) pairs in validation dataset
            validation_error = math.sqrt(validation_evaluation_RDD.map(lambda z: (z[1][0] - z[1][1])**2).mean())
            # Save the error as a row in a pandas DataFrame
            hyperparams_eval_df.loc[index] = [k, regularization, iterations, validation_error, float('inf')]
            index = index + 1
            # Check whether the current error is the lowest
            if validation_error < lowest_validation_error:
                best_k = k
                best_regularization = regularization
                best_iterations = iterations
                best_index = index - 1
                lowest_validation_error = validation_error
print('The best rank k is ', best_k, ', regularization = ', best_regularization, ', iterations = ',      best_iterations, '. Validation Error =', lowest_validation_error)


# # Use Testing Data to Evaluate the Model built using the Best Hyperparameters                

# # Evaluating the best hyperparameter combination using testing data

# In[36]:


seed = 37
model = ALS.train(training_RDD, best_k, seed=seed, iterations=best_iterations, lambda_=best_regularization)
testing_prediction_RDD=model.predictAll(testing_input_RDD).map(lambda x: ((x[0], x[1]), x[2]))
testing_evaluation_RDD= test_RDD.map(lambda x: ((x[0], x[1]), x[2])).join(testing_prediction_RDD)
testing_error = math.sqrt(testing_evaluation_RDD.map(lambda x: (x[1][0]-x[1][1])**2).mean())
print('The Testing Error for rank k =', best_k, ' regularization = ', best_regularization, ', iterations = ',       best_iterations, ' is : ', testing_error)


# In[37]:


#print(best_index)


# In[38]:


# Store the Testing RMS in the DataFrame
hyperparams_eval_df.loc[best_index]=[best_k, best_regularization, best_iterations, lowest_validation_error, testing_error]


# In[39]:


schema3= StructType([ StructField("k", FloatType(), True),                       StructField("regularization", FloatType(), True ),                       StructField("iterations", FloatType(), True),                       StructField("Validation RMS", FloatType(), True),                       StructField("Testing RMS", FloatType(), True)                     ])


# ## Converting the pandas DataFrame that stores validation errors of all hyperparameters and the testing error for the best model to a Spark DataFrame, so that it can be written in the cluster mode.

# In[40]:


HyperParams_RMS_DF = ss.createDataFrame(hyperparams_eval_df, schema3)


# ## Output Path

# In[41]:


output_path = "/storage/home/ALSHyperParamsTuning"
HyperParams_RMS_DF.write.option("header", True).csv(output_path)


# In[42]:


ss.stop()


# In[ ]:




