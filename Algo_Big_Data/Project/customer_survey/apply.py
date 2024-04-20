import os
import sys
import pandas as pd
import numpy as np
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(dirname)))
from exploitation import Exploitation
from models import Models
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

sc = SparkContext.getOrCreate()

spark = SparkSession(sc)
class Apply:
    def __init__(self, data):
        self.data = data
        
    def run(self):
        #Clean Dataset and split data
        expl = Exploitation(self.data)
        self.train, self.test =  expl.run()
        #Apply Models
        models = Models(self.train, self.test)
        print("-------------------LOGISTIC REGRESTION---------------")
        models.logistic_regression()
        print("-------------------DECISION TREE---------------")
        models.decision_tree()
        print("-------------------RANDOM FOREST---------------")
        models.random_forest()
        print("-------------------GRADIENT BOOSTING---------------")
        models.gradient_boosting()
        print("-------------------SUPPORT VECTOR MACHINE---------------")
        models.linear_support_vector_machine()
        
df = spark.read.csv("bank_customer_survey.csv", header = True, inferSchema = True)
df.printSchema()
apply = Apply(df)
apply.run()

