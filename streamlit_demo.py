import streamlit as st
import pandas as pd
import numpy as np

st.title("\"Sure or Not?\": Fake News Detection Using Data Mining Approaches")
st.subheader("IS434: Data Mining and Business Analytics")
st.markdown("With the **evolution of news media** - digital forms of online news, blogs and social media instead of traditional forms of news from newspapers, tabloids and magazines, and the **rise in spread of fake news**, we want to create a model that can **detect fake news** to ensure that people are aware of the credibility of news they see online.")

text = st.text_area("Enter article text here!")
st.button("Predict")
    
# output = predict(text)