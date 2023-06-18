
import shap
from shap import TreeExplainer
import streamlit as st
import streamlit.components.v1 as components
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

    
# Set page title
st.set_page_config(
    page_title="France Road Accident Data Analysis - SHAP ",
    layout='wide'
)

st.markdown(f'<b><h0 style="color:#00008B;font-size:35px;">{"Model interpretation with SHAP:"}</h0><br>', unsafe_allow_html=True)

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

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

df = load_data('Datasets/X_test_sample.csv')
df = df.sample(n=1000)




"""
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

             
y_test =df['AccidentSeverity']
X_test = df.drop(['AccidentSeverity'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model = xgb.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
    
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
"""
y_test =df['AccidentSeverity']
X_test = df.drop(['AccidentSeverity'], axis = 1)
xgb_imp = xgb.XGBClassifier()
model = xgb_imp.load_model('Models/xgb_model_improved.json')

explainer = TreeExplainer(xgb_imp) # XGBoost Classifier Improved
# Compute shap values
shap_values = explainer.shap_values(X_test[0:100])
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;"><br><b>{"Visualize the first prediction explanation"}</p>', unsafe_allow_html=True)
# Initialize shap
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], features=X_test.iloc[0], feature_names=X_test.columns)
#st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]))
#st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;"><br><b>{"Visualize the test set predictions"}</p>', unsafe_allow_html=True)
#st_shap(shap.force_plot(explainer.expected_value, shap_values, X), 400)
