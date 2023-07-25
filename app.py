import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
from fuctions.spotify import *

import streamlit as st
import pandas as pd
import plotly.express as px
import plost


st.set_page_config(layout='wide', initial_sidebar_state='expanded')


st.sidebar.markdown('''
---
Created with ❤️ by [Data Professor](https://youtube.com/dataprofessor/).
''')


# Row A
st.markdown('### Metrics')
col1, col2, col3 = st.columns(3)
col1.metric(f"Seconds Listeng Music", str(time_songs(1000)) + ' seg')
col2.metric(f"Minutes Listeng Music", str(time_songs(60000)) + ' min')
col3.metric(f"Hours Listeng Music", str(time_songs(3600000)) + ' hours')

# Row B
seattle_weather = pd.read_csv('https://raw.githubusercontent.com/tvst/plost/master/data/seattle-weather.csv', parse_dates=['date'])
stocks = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/stocks_toy.csv')

values = st.dataframe(tracks_dataframe())

df = count_artist()
df_tracks = count_tracks()

c1, c2 = st.columns((5,5))
with c1:
    st.markdown('### Bar chart')
    fig = px.bar(df_tracks, y=df_tracks['Tracks Names'], x=df_tracks['index'])
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.markdown('### Donut chart')
    fig = px.pie(df, values=df['Artist Names'], names=df['index'])
    st.plotly_chart(fig, use_container_width=True)

# Row C

st.markdown('### Recomendation Tracks')
for i, index in enumerate(create_recommended_playlist()):
    st.write(i, index)




