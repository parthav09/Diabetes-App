import streamlit as st
import pandas as pd
import pickle

st.title('Diabetes Application')
global submitted
with st.form('Enter the details below'):
    st.write('Enter the number of pregnancies')
    pregnancies = st.slider(label="Drag the Slider Below", min_value=0, max_value=20, key='pregnant')

    st.write('Enter the glucose level')
    glucose = st.slider(label="Drag the Slider Below", min_value=0, max_value=2900, key='glucose')

    st.write('Enter the Blood pressure level')
    bp = st.slider(label="Drag the Slider Below", min_value=0, max_value=2900, key='bp')

    st.write('Enter your Skin thickness')
    thickness = st.slider(label="Drag the Slider Below", min_value=0, max_value=50, key='thickness')

    st.write('Enter the Insulin level')
    insulin = st.slider(label="Drag the Slider Below", min_value=0, max_value=400, key='insulin')

    st.write('Enter the BMI level')
    bmi = st.slider(label="Drag the Slider Below", min_value=0.0, max_value=80.0, key='BMI')

    st.write('Enter the DiabetesPedigreeFunction')
    pedi = st.slider(label="Drag the Slider Below", min_value=0.000, max_value=1.000, key='pedi')

    st.write('Enter the age')
    age = st.slider(label="Drag the Slider Below", min_value=0, max_value=120, key='age')

    submitted = st.form_submit_button("Submit")

if submitted:
    data = [[pregnancies, glucose, bp, thickness, insulin, bmi, pedi, age]]
    X = pd.DataFrame(data,
                     columns=['pregnancies', 'glucose', 'bp', 'thickness', 'insulin', 'bmi', 'pedi', 'age'])
    pickle_in = open('model.pkl', 'rb')
    model = pickle.load(pickle_in)
    result = model.predict(X)

    if result == 1:
        st.header("You have been suffering through **diabetes**.")
    else:
        st.header("Have some chocolates.")
