import shap
import streamlit as st
import streamlit.components.v1 as components
import xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
    

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
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

             
y =df['AccidentSeverity']
X = df.drop(['AccidentSeverity'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

st.markdown(f'<b><h0 style="color:#00008B;font-size:35px;">{"Model Interpretation with SHAP(SHapley Additive exPlanations) :"}</h0><br>', unsafe_allow_html=True)
    
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;"><b>{"Visualize the first prediction explanation:"}</b></p>', unsafe_allow_html=True)
st_shap(shap.force_plot(explainer.expected_value, shap_values[3,:], X.iloc[3,:]))
st.write(X.iloc[3,:])
st.write(y[3])
st.write(shap_values[3])
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;"><b>{"Visualize the predictions:"}</b></p>', unsafe_allow_html=True)
st_shap(shap.force_plot(explainer.expected_value, shap_values, X), 400)
