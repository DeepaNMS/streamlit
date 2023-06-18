
### This is the data modelling subpage of streamlit webapp

import streamlit as st
import pandas as pd
import numpy as np

# Set page title
st.set_page_config(
    page_title="France Road Accident Data Analysis - Modelling",
    layout='wide'
)

st.markdown(f'<b><h0 style="color:#00008B;font-size:35px;">{"Data Modelling:"}</h0><br><br>', unsafe_allow_html=True)

# Content of the page
option = st.selectbox(
    'Choose a data modelling result criteria: ',
    ('Accuracy Score', 
     'Training Time', 
     'Precision',
     'Recall',
     'F1 Score'
    ))
    
if option == 'Accuracy Score':
    dict = {'Model Name':['XGBoost', 'Gradient Boost', 'Random Forest', 'AdaBoost'],
            'Training set score':[71.99, 71.89, 91.1, 69.5],
            'Test set score':[71.4, 71.4, 66.7, 69.7]
       }
    df_AccuracyScore = pd.DataFrame(dict)
    st.table(df_AccuracyScore)

elif option == 'Training Time':
    dict = {'Model Name':['XGBoost', 'Gradient Boost', 'Random Forest', 'AdaBoost'],
            'Training Time (sec)':[63.92, 404.98, 112.30, 28.21]
           }
    df_TrainingTime = pd.DataFrame(dict)
    st.table(df_TrainingTime)
    
elif option == 'Precision':
    dict = {'Model Name':['XGBoost', 'Gradient Boost', 'Random Forest', 'AdaBoost'],
            'Precision - Class 0 (Not injured/Slightly injured) ':[74,74,72,72],
            'Precision - Class 1 (Heavily injured/Died) ':[65,65,56,64]
           }
    df_Precision = pd.DataFrame(dict)
    st.table(df_Precision)
    
elif option == 'Recall':
    dict = {'Model Name':['XGBoost', 'Gradient Boost', 'Random Forest', 'AdaBoost'],
            'Recall - Class 0 (Not injured/Slightly injured) ':[85,85,78,86],
            'Recall - Class 1 (Heavily injured/Died) ':[48,48,47,42]
           }
    df_Recall = pd.DataFrame(dict)
    st.table(df_Recall)

elif option == 'F1 Score':
    dict = {'Model Name':['XGBoost', 'Gradient Boost', 'Random Forest', 'AdaBoost'],
            'F1 Score - Class 0 (Not injured/Slightly injured) ':[79,79,75,78],
            'F1 Score - Class 1 (Heavily injured/Died) ':[55,56,51,50]
           }
    df_F1Score = pd.DataFrame(dict)
    st.table(df_F1Score)

    
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
# List down all features used for modelling
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;"><br><b>{"Accident features used in data modelling!"}</p><br>', unsafe_allow_html=True)
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;">{"1. Area Zone "}</p><br>', unsafe_allow_html=True)
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;">{"2. Collision Type"}</p><br>', unsafe_allow_html=True)
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;">{"3. Municipality "}</p><br>', unsafe_allow_html=True)
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;"><br>{"4. Road Category "}</p><br>', unsafe_allow_html=True)
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;"><br>{"5. Traffic Regime "}</p><br>', unsafe_allow_html=True)
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;"><br>{"6. Nr Of Traffic Lanes "}</p><br>', unsafe_allow_html=True)
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;"><br>{"7. Accident Situation "}</p><br>', unsafe_allow_html=True)
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;"><br>{"8. Struck With Fixed Object "}</p><br>', unsafe_allow_html=True)
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;"><br>{"9. Struck With Moving Object "}</p><br>', unsafe_allow_html=True)
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;"><br>{"10.Initial Shock Point "}</p><br>', unsafe_allow_html=True)
st.markdown(f'<p align="justify" font-family: "Times New Roman" style="color:#000000;"><br>{"11.Intersection Type "}</p><br>', unsafe_allow_html=True)
