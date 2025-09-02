import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import base64

def set_background_image(image_path):
    try:
        with open(image_path, "rb") as f:
            img_data = f.read()
        b64_encoded = base64.b64encode(img_data).decode()
        
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/png;base64,{b64_encoded});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            
            /* Optional: Add semi-transparent overlay for better text readability */
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(255, 255, 255, 0.1);
                z-index: -1;
            }}
            
            /* Make containers slightly transparent for better blend */
            .element-container {{
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 10px;
                padding: 10px;
                margin: 5px 0;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("Background image 'img.jpg' not found. Please make sure the file exists in the same directory as your script.")

set_background_image("img.jpg")

st.title('Palmer Penguins Classification Model')
st.info("🐧 Discover which penguin species matches your measurements! Explore the fascinating world of Antarctic wildlife through AI-powered classification.")

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
  df 
  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  X_raw
  st.write('**y**') 
  y_raw = df.species
  y_raw

with st.expander('Data Visualization'):
  st.scatter_chart(data = df , x = 'bill_length_mm' , y = 'body_mass_g' , color = 'species' )

# input features
with st.sidebar:
  st.header('Input Features')
  island = st.selectbox('island', ('Torgersen', 'Biscoe', 'Dream') )
  sex = st.selectbox('sex', ('male', 'female'))
  bill_length_mm = st.slider('bill_length_mm', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('bill_depth_mm', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('flipper_length_mm', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('body_mass_g', 2700.0, 6300.0, 4207.0 )

# create a dataframe for the input features.
  data = {
    'island': island,
    'bill_length_mm': bill_length_mm,
    'bill_depth_mm': bill_depth_mm,
    'flipper_length_mm': flipper_length_mm,
    'body_mass_g': body_mass_g,
    'sex': sex}
  input_df = pd.DataFrame(data, index = [0])
  input_penguins = pd.concat([input_df,X_raw], axis=0)

with st.expander('input features'):
  st.write('**input penguin**')
  input_df
  st.write('**combined penguin data**')
  input_penguins

# Data Preparation
# Encode 
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)
X = df_penguins[1:]
input_row =df_penguins[:1]

# Encode y
target_mapper = {'Adelie':0,
                  'Chinstrap': 1,
                  'Gentoo': 2}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write("**Encoded X (input penguin)**")
  input_row
  st.write('**Encoded y**')   
  y

# Model training and inference
## Model training
clf = RandomForestClassifier()
clf.fit(X, y)

## Apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_proba.rename(columns={0: 'Adelie',
                                    1: 'Chinstrap',
                                    2: 'Gentoo'})
df_prediction_proba                          

# Display predicted species
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba,
             column_config={
               'Adelie': st.column_config.ProgressColumn(
                 'Adelie',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               
               'Chinstrap': st.column_config.ProgressColumn(
                 'Chinstrap',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               
               'Gentoo': st.column_config.ProgressColumn(
                 'Gentoo',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               )}, hide_index=True)

penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction][0]))
