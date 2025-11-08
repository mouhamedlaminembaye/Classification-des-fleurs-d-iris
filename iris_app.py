import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()

iris.data

iris.target 

rfc = RandomForestClassifier()
rfc.fit(iris.data, iris.target)



