from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import joblib
import sklearn
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics


def results1(model):
    st.markdown('## Accuracy')
    y_pred = model.predict(X_test)
    res = (accuracy_score(y_test,y_pred)) * 100
    st.write('Accuracy score for test dataset =', res, '%')   
    st.markdown('## Confusion Matrix')
    st.dataframe(confusion_matrix(y_test, y_pred))
    st.markdown('## Classification Report')
    st.text(classification_report(y_test, y_pred))

st.markdown('# Interactive Data Modelling:')
# To set the background image of the page
st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/free-photo/abstract-luxury-gradient-blue-background-smooth-dark-blue-with-black-vignette-studio-banner_1258-63452.jpg?size=626&ext=jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;"><br>{"Generally speaking we can consider that accuracy scores:"}</p><br>', unsafe_allow_html=True)
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;"><br>{"Over 90% - Very good"}</p><br>', unsafe_allow_html=True)
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;"><br>{"Between 70% and 90% - Good"}</p><br>', unsafe_allow_html=True)
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;"><br>{"Between 60% and 70% - OK"}</p><br>', unsafe_allow_html=True)

choices = ['XGBOOST','XGBOOST Improved']
option = st.selectbox('Which model do you want to try ?', choices)

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

df = load_data('Datasets/X_test.csv')
df1 = load_data('Datasets/y_test.csv')

if df is not None:
    y_test =df1['AccidentSeverity']
    X_test = df

if option=='XGBOOST':
   st.write('Accuracy score for training dataset = 71.998%')
   xgb = xgb.XGBClassifier()
   xgb.load_model('Models/xgb_model.json')
   results1(xgb)
    
elif option=='XGBOOST Improved':
   st.write('XGBOOST score train 72.052')
   xgb_imp = xgb.XGBClassifier()
   xgb_imp.load_model('Models/xgb_model_improved.json')
   results1(xgb_imp) 
