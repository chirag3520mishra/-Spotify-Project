from turtle import st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import streamlit as st

st.markdown('<h2 style="color: maroon; text-align:center; background-color: white; font-family: verdana; border: 3px dashed #028A0F; border-radius:5px; margin-bottom: 20px; ">Spotify Songs Data Visualisation</h2>', unsafe_allow_html=True)

st.markdown('''<p style="font-weight:bold; font-size: 20px; font-family:Cursive"> Unlock the secrets of music popularity on Spotify with Beatstats! Discover the surprising 
             truth about editorial playlists, the power of context in music discovery, and the ultimate guide to growing your fanbase. Get instant access to expert insights
             and data-driven strategies to boost your music career. </p>''', unsafe_allow_html=True)
st.image('assets\images.png', use_column_width=True)

st.markdown("<h1 style='text-align: center; font-family: Times New Roman, serif;'><b>About The Project :</b></h1>", unsafe_allow_html=True)
st.markdown('''<p style="font-weight:bold; font-size: 20px; font-family:Cursive"> In this project, we aim to provide a comprehensive dataset of Spotify tracks, encompassing a diverse range of 125 genres. The dataset includes 17 columns,
             each representing a unique audio feature associated with individual tracks. By analyzing this dataset, we can gain insights into the musical characteristics
             of each track, identify patterns across different genres, and develop machine learning models to predict genres based on audio features.</p>''', unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; font-family: Times New Roman, serif;'><b>About The Dataset :</b></h1>", unsafe_allow_html=True)
st.markdown('''<p style="font-weight:bold; font-size: 20px; font-family:Cursive"> This dataset contains information about Spotify tracks, with 17 columns that describe different aspects of each track. The columns include the 
            artist and album names, track name, popularity score, duration, and whether the track contains explicit content. There are also columns that measure
            the track's danceability, energy, loudness, and other musical characteristics. Additionally, the dataset includes columns that indicate the presence
            of spoken words, acoustic quality, instrumentalness, and audience presence during recording. Finally, the dataset includes the track's tempo, time 
            signature, and genre.</p>''', unsafe_allow_html=True)
