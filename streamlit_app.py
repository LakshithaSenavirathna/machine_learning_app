import streamlit as st
import pandas as pd

st.title('Machine Learning App')

st.info("Let's make a machine learning model.")

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
  df 

  st.write('**X**')
  X = df.drop('species', axis=1)
  X

  st.write('**y**') 
  y = df.species
  y

with st.expander('Data Visualization'):
  st.scatter_chart(data = df , x = 'bill_length_mm' , y = 'body_mass_g' , color = 'species' )


# data preparation

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
  input_penguins = pd.concat([input_df,X], axis=0)
  
  # Encode 
  encode = ['island', 'sex']
  df_penguins = pd.get_dummies(input_penguins, prefix=encode)
  input_row =df_penguins[:1]

with st.expander('input features'):
  st.write('**input penguin**')
  input_df
  st.write('**combined penguin data**')
  input_penguins
  st.write("Encoded input penguin')
  input_row
  

