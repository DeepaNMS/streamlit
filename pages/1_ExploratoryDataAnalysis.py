### This is the exploratory data analysis subpage of streamlit webapp 
    
    
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling

# Set page title
st.set_page_config(
    page_title="France Road Accident Data Analysis - EDA ",
    layout='wide'
)

st.markdown(f'<b><h0 style="color:#00008B;font-size:35px;">{"Exploratory Data Analysis:"}</h0><br>', unsafe_allow_html=True)

# To insert textual content 
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;"><br>{"Since the dataset is huge, the profiling has been done on the Test dataset."}</p><br>', unsafe_allow_html=True)

    
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
# This page deals with the exploratory data analysis 

X_test_file = "Datasets/data_viz_plot_filtered.csv"
df = pd.read_csv(X_test_file).sample(n=25000)
pr = df.profile_report()
st_profile_report(pr)
