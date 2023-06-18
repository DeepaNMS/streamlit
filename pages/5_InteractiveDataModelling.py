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

def results(model):
    st.markdown('## Accuracy')
    st.write(model.score(X_test, y_test))
    st.markdown('## Confusion Matrix')
    st.dataframe(confusion_matrix(y_test, model.predict(X_test)))
    st.markdown('## Classification Report')
    st.text(classification_report(y_test, model.predict(X_test)))

def results1(model):
    st.markdown('## Accuracy')
    y_pred = model.predict(X_test)
    st.write(accuracy_score(y_test,y_pred))
    st.markdown('## Confusion Matrix')
    st.dataframe(confusion_matrix(y_test, y_pred))
    st.markdown('## Classification Report')
    st.text(classification_report(y_test, y_pred))

def splitDataset(df):
   y_test =df['severity']
   X_test = df.drop(['severity'], axis = 1)
   return y_test,X_test
   


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

st.write("""Generally speaking we can consider that accuracy scores:
                          - Over 90% - Very Good
                    - Between 70% and 90% - Good
                    - Between 60% and 70% - OK""")

choices = ['XGBOOST','XGBOOST Improved']
option = st.selectbox('Which model do you want to try ?', choices)

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

df = load_data('Datasets/X_test.csv')
df1 = load_data('Datasets/y_test.csv')

st.write(df.columns)
st.write(df1.columns)

if df is not None:
    y_test =df1['AccidentSeverity']
    X_test = df

if option=='XGBOOST':
   st.write('XGBOOST score train 71.998')
   xgb = xgb.XGBClassifier()
   xgb.load_model('Models/xgb_model.json')
   results1(xgb)


if option=='XGBOOST Improved':
   st.write('XGBOOST score train 72.052')
   xgb_imp = xgb.XGBClassifier()
   xgb_imp.load_model('Models/xgb_model_improved.json')
   results1(xgb_imp) 
