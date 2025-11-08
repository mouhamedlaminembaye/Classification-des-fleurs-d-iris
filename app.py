import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.title("""Iris Classification App""")
st.write(""" Une simple App pour la prediction des fleurs d'iris.
         L'applicaton predit la categorie des fleurs d'iris""")

st.sidebar.header(" Les parametrs d'entree")

def user_input():
    sepal_length = st.sidebar.slider("Longueur du sepal", 4.3, 7.9, 5.3 )
    sepal_width = st.sidebar.slider("Largeur du sepal", 2.0, 4.4, 3.3)
    petal_length = st.sidebar.slider("Longueur du petal", 1.0, 6.9, 2.3)
    petal_width = st.sidebar.slider("Largeur du petal", 0.1, 2.5, 1.3)
    
    data = {
        'sepal_length' : sepal_length,
        'sepal_width' : sepal_width,
        'petal_length' : petal_length,
        'petal_width' : petal_width
    }
    
    parametres_fleurs = pd.DataFrame(data, index=[0])
    return parametres_fleurs

df = user_input()
st.subheader("On veut trouver la categorie de cette fleur")
st.write(df)
iris = datasets.load_iris()
clf = RandomForestClassifier()
clf.fit(iris.data, iris.target)

prediction = clf.predict(df)
st.subheader("La categorie de la fleur est :")
st.write( iris.target_names[prediction])