
### This is the prediction subpage of streamlit webapp
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier

# Set page title
st.set_page_config(
    page_title="France Road Accident Data Analysis - Prediction",
    layout='wide'
)

st.markdown(f'<b><h0 style="color:#00008B;font-size:35px;">{"Prediction:"}</h0><br><br>', unsafe_allow_html=True)

# Content of the page
###### AREAZONE
strareaZone = st.selectbox(
                        'Area Zone :',
                        ('Outside agglomeration', 'In built up areas'))

if strareaZone == 'Outside agglomeration' :
   areaZone = 1
elif strareaZone == 'In built up areas':
   areaZone = 2

###### COLLISSIONTYPE
strcollissionType = st.selectbox(
                        'Collision Type :',
                        ('Two vehicles frontal','Two vehicles from behind','Two vehicles from the side',
                         'Three or more vehicles in a chain','Multiple collision','Other collision','Collision free'))

if strcollissionType == 'Two vehicles frontal' :
   collisionType = 1
if strcollissionType == 'Two vehicles from behind' :
   collisionType = 2
if strcollissionType == 'Two vehicles from the side' :
   collisionType = 3
if strcollissionType == 'Three or more vehicles in a chain' :
   collisionType = 4
if strcollissionType == 'Multiple collision' :
   collisionType = 5
if strcollissionType == 'Other collision' :
   collisionType = 6
if strcollissionType == 'Collision free' :
   collisionType = 7

###### MUNICIPALITY
municipality = st.number_input("Municipality")

###### ROADCATEGORY
strroadCategory = st.selectbox(
                        'Road Category :',
                        ('Motorway','National Road','Departmental Road','Communal Road','Outside the public network',
                         'Car park open to publc traffic','Urban Metropolis Routes','Other'))

if strroadCategory == 'Motorway' :
   roadCategory = 1
if strroadCategory == 'National Road' :
   roadCategory = 2
if strroadCategory == 'Departmental Road' :
   roadCategory = 3
if strroadCategory == 'Communal Road' :
   roadCategory = 4
if strroadCategory == 'Outside the public network' :
   roadCategory = 5
if strroadCategory == 'Car park open to publc traffic' :
   roadCategory = 6
if strroadCategory == 'Urban Metropolis Routes' :
   roadCategory = 7
if strroadCategory == 'Other' :
   roadCategory = 9

##### TRAFFICREGFIME
strtrafficRegime = st.selectbox(
                        'Traffic Regime :',
                        ('None','One way','Bidirectional','With separate carriageways','With variable assignment channels'))

if strtrafficRegime == 'None' :
   trafficRegime = -1
if strtrafficRegime == 'One way' :
   trafficRegime = 1
if strtrafficRegime == 'Bidirectional' :
   trafficRegime = 2
if strtrafficRegime == 'With separate carriageways' :
   trafficRegime = 3
if strtrafficRegime == 'With variable assignment channels' :
   trafficRegime = 4


##### NROFTRAFFICLANES
nrOfTrafficLanes = st.number_input("Number Of Traffic Lanes")

##### ACCIDENTSITUATION
straccidentSituation = st.selectbox(
                        'Accident Situation :',
                        ('None','On the road','On hard shoulder','On shoulder','On sidewalk','On a cycle path','On another special lane','Others'))

if straccidentSituation == 'None' :
   accidentSituation = 0
if straccidentSituation == 'On the road' :
   accidentSituation = 1
if straccidentSituation == 'On hard shoulder' :
   accidentSituation = 2
if straccidentSituation == 'On shoulder' :
   accidentSituation = 3
if straccidentSituation == 'On sidewalk' :
   accidentSituation = 4
if straccidentSituation == 'On a cycle path' :
   accidentSituation = 5
if straccidentSituation == 'On another special lane' :
   accidentSituation = 6
if straccidentSituation == 'Others' :
   accidentSituation = 8

##### STRUCKWITHFIXEDOBJECT
strstruckWithFixedObject = st.selectbox(
                        'Struck With Fixed Object :',
                        ('None','Parked vehicle','Tree','Metal slider','Concrete slide','Other slide','Building, wall, bridge pier',
                         'Vertical signaling support or emergency call station','Post','Street furniture','Parapet','Island, refuge, high boundary',
                         'Sidewalk curb','Ditch, embankment, rock wall','Other fixed obstacle on roadway','Other fixed obstacle on sidewalk or shoulder',
                         'Obstacle-free road exit','Nozzle – aqueduct head'))

if strstruckWithFixedObject == 'None' :
   struckWithFixedObject = 0
elif strstruckWithFixedObject == 'Parked vehicle' :
   struckWithFixedObject = 1
elif strstruckWithFixedObject == 'Tree' :
   struckWithFixedObject = 2
elif strstruckWithFixedObject == 'Metal slider' :
   struckWithFixedObject = 3
elif strstruckWithFixedObject == 'Concrete slide' :
   struckWithFixedObject = 4
elif strstruckWithFixedObject == 'Other slide' :
   struckWithFixedObject = 5
elif strstruckWithFixedObject == 'Building, wall, bridge pier' :
   struckWithFixedObject = 6
elif strstruckWithFixedObject == 'Vertical signaling support or emergency call station' :
   struckWithFixedObject = 7
elif strstruckWithFixedObject == 'Post' :
   struckWithFixedObject = 8
elif strstruckWithFixedObject == 'Street furniture' :
   struckWithFixedObject = 9
elif strstruckWithFixedObject == 'Parapet' :
   struckWithFixedObject = 10
elif strstruckWithFixedObject == 'Island, refuge, high boundary' :
   struckWithFixedObject = 11
elif strstruckWithFixedObject == 'Sidewalk curb' :
   struckWithFixedObject = 12
elif strstruckWithFixedObject == 'Ditch, embankment, rock wall' :
   struckWithFixedObject = 13
elif strstruckWithFixedObject == 'Other fixed obstacle on roadway' :
   struckWithFixedObject = 14
elif strstruckWithFixedObject == 'Other fixed obstacle on sidewalk or shoulder' :
   struckWithFixedObject = 15
elif strstruckWithFixedObject == 'Obstacle-free road exit' :
   struckWithFixedObject = 16
elif strstruckWithFixedObject == 'Nozzle – aqueduct head' :
   struckWithFixedObject = 17


##### STRUCKWITHMOVINGOBJECT
strstruckWithMovingObject = st.selectbox(
                        'Struck With Moving Object :',
                        ('None','Pedestrian','Vehicle','Rail vehicle','Pet','Wild animal','Other'))

if strstruckWithMovingObject == 'None' :
   struckWithMovingObject = 0
if strstruckWithMovingObject == 'Pedestrian' :
   struckWithMovingObject = 1
if strstruckWithMovingObject == 'Vehicle' :
   struckWithMovingObject = 2
if strstruckWithMovingObject == 'Rail vehicle' :
   struckWithMovingObject = 4
if strstruckWithMovingObject == 'Pet' :
   struckWithMovingObject = 5
if strstruckWithMovingObject == 'Wild animal' :
   struckWithMovingObject = 6
if strstruckWithMovingObject == 'Other' :
   struckWithMovingObject = 9


##### INITIALSHOCKPOINT
strinitialShockPoint = st.selectbox(
                        'Initial Shock Point :',
                        ('None','Before','Front right','Front left','Back','Right back','Left Rear','Right side','Left side','Multiple shocks (barrels)'))

if strinitialShockPoint == 'None' :
   initialShockPoint = 0
if strinitialShockPoint == 'Before' :
   initialShockPoint = 1
if strinitialShockPoint == 'Front right' :
   initialShockPoint = 2
if strinitialShockPoint == 'Front left' :
   initialShockPoint = 3
if strinitialShockPoint == 'Back' :
   initialShockPoint = 4
if strinitialShockPoint == 'Right back' :
   initialShockPoint = 5
if strinitialShockPoint == 'Left Rear' :
   initialShockPoint = 6
if strinitialShockPoint == 'Right side' :
   initialShockPoint = 7
if strinitialShockPoint == 'Left side' :
   initialShockPoint = 8
if strinitialShockPoint == 'Multiple shocks (barrels)' :
   initialShockPoint = 9

##### INTERSECTIONTYPE
strintersectionType = st.selectbox(
                        'Intersection Type :',
                        ('Out of intersection','X intersection','T-junction','Y intersection','Intersection with more than 4 branches',
                         'Roundabout','Square','Level crossing','Other intersection'))

if strintersectionType == 'Out of intersection' :
   intersectionType = 1
if strintersectionType == 'X intersection' :
   intersectionType = 2
if strintersectionType == 'T-junction' :
   intersectionType = 3
if strintersectionType == 'Y intersection' :
   intersectionType = 4
if strintersectionType == 'Intersection with more than 4 branches' :
   intersectionType = 5
if strintersectionType == 'Roundabout' :
   intersectionType = 6
if strintersectionType == 'Square' :
   intersectionType = 7
if strintersectionType == 'Level crossing' :
   intersectionType = 8
if strintersectionType == 'Other intersection' :
   intersectionType = 9


#Loading up the XGBoost model we created

xgb_cl = xgb.XGBClassifier()
xgb_cl.load_model('Models/xgb_model.json')


if st.button('Predict Accident Severity'):
    X = pd.DataFrame([[areaZone, collisionType, municipality, roadCategory, trafficRegime, nrOfTrafficLanes, accidentSituation, struckWithFixedObject, struckWithMovingObject, initialShockPoint, intersectionType ]], columns=[areaZone, collisionType, municipality, roadCategory, trafficRegime, nrOfTrafficLanes, accidentSituation, struckWithFixedObject, struckWithMovingObject, initialShockPoint, intersectionType]) 
    prediction = xgb_cl.predict(X)
    if prediction == 0:
        st.write('Class 0 - Not Injured/Slightly injured')
    elif prediction == 1:
        st.write('Class 1 - Heavily Injured/Died')
    prob = model.predict_proba(X)
    st.write(prob)
   
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
