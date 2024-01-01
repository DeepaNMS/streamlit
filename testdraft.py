### This is the main page of the web app ###
import streamlit as st

# Set page title
st.set_page_config(
    page_title="France Road Accident Data Analysis",
    layout='wide'
)

# Content of the page
st.markdown(f'<b><h0 style="color:#000000;font-size:35px;">{"France Road Accidents Data Analysis & Severity Prediction !"}</h0><br>', unsafe_allow_html=True)

# To insert image
st.image("https://upload.wikimedia.org/wikipedia/commons/2/2f/Multi_vehicle_accident_-_M4_Motorway%2C_Sydney%2C_NSW_%288076208846%29.jpg",
            width=900 # Manually Adjust the width of the image as per requirement
        )
    
# To insert textual content 
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;">{"The objective of this project is to try to predict the severity of road accidents in France using historical data. By analyzing past records, the aim is to develop a predictive model that can estimate the severity of accidents. This project presents an ideal opportunity to encompass all stages of a comprehensive Data Science project. "}</p><br>', unsafe_allow_html=True)
