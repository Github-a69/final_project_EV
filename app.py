

import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
import category_encoders 
from sklearn.preprocessing import RobustScaler
from category_encoders import BinaryEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline



st.set_page_config(layout='wide',page_title='electric cars')



# Design system 

st.markdown("""
<style>

/* =========================
   GOOGLE FONT
========================= */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* =========================
   DARK THEME
========================= */
.stApp {
    background: #0B1120 !important;
}

/* =========================
   SIDEBAR
========================= */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #1e1b4b,
        #111827
    ) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Sidebar headings */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* =========================
   METRIC CARDS
========================= */
[data-testid="metric-container"] {
    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed,
        #ec4899
    ) !important;

    border-radius: 18px !important;
    padding: 18px !important;
    margin-bottom: 14px !important;

    box-shadow:
        0 4px 20px rgba(124,58,237,.35),
        0 0 25px rgba(236,72,153,.15);

    transition: all 0.3s ease;
}

/* Hover effect */
[data-testid="metric-container"]:hover {
    transform: translateY(-4px);
}

/* Metric title */
[data-testid="stMetricLabel"] {
    color: #F8FAFC !important;
    font-size: 15px !important;
    font-weight: 800 !important;
    letter-spacing: .5px !important;
    text-transform: uppercase !important;
}

/* Metric value */
[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 34px !important;
    font-weight: 900 !important;
}

/* Delta */
[data-testid="stMetricDelta"] {
    color: #86EFAC !important;
    font-weight: 700 !important;
}

/* =========================
   BUTTONS
========================= */
.stButton > button {
    background: linear-gradient(
        135deg,
        #06b6d4,
        #2563eb
    ) !important;

    color: white !important;
    border: none !important;

    border-radius: 10px !important;
    font-weight: 700 !important;

    transition: all .3s ease;
}

.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 15px rgba(37,99,235,.5);
}

/* =========================
   SELECTBOX
========================= */
.stSelectbox div[data-baseweb="select"] {
    background-color: #111827 !important;
    border-radius: 10px !important;
}

/* =========================
   RADIO BUTTONS
========================= */
div[role="radiogroup"] label {
    color: white !important;
    font-weight: 500 !important;
}

/* =========================
   TABS
========================= */
.stTabs [data-baseweb="tab"] {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 3px solid #38bdf8 !important;
}

/* =========================
   CHART CONTAINERS
========================= */
[data-testid="stPlotlyChart"] {
    background: #111827 !important;
    border-radius: 15px !important;
    padding: 10px !important;
    box-shadow: 0 0 12px rgba(0,0,0,.3);
}

/* =========================
   HEADINGS
========================= */
h1 {
    color: #38bdf8 !important;
    font-weight: 800 !important;
}

h2 {
    color: #a78bfa !important;
    font-weight: 700 !important;
}

h3 {
    color: #f472b6 !important;
    font-weight: 700 !important;
}

</style>
""", unsafe_allow_html=True)










df=pd.read_csv('clean_electric_vehicle.csv')



# importing random forest pipeline
model = joblib.load('best_model.pkl')

# Dividing the webage into multiple sections

page=st.sidebar.radio('choose an option',['Overview','Analysis','Prediction'])


if page == 'Overview':

    st.image('https://images.unsplash.com/photo-1593941707874-ef25b8b4a92b?q=80&w=1172&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')
    st.dataframe(df)
    # explaining features meaning
    cols = {'model year':'The manufacturing/model year of the vehicle',
            'make':'The vehicle manufacturer / brand',
            'model':'The model name of the vehicle' , 
            'electric vehicle type':' BEV (Battery Electric Vehicle) — runs purely on electric battery no combustion engine.\n  PHEV (Plug-in Hybrid Electric Vehicle) — has both an electric motor and a gasoline engine',
            'clean alternative fuel vehicle (cafv) eligibility':'Whether the vehicle qualifies  CAFV program (tax/incentive benefits).\n Three possible values:Clean Alternative Fuel Vehicle Eligible — meets the requirements. Not eligible due to low battery range — battery range is too short to qualify. Eligibility unknown as battery range has not been researched — insufficient data to determine',
            'electric range':"The vehicle's all-electric driving range in miles",
            'legislative district':'legislative district number where the vehicle is registered — used for policy/representation purposes',
            'number of utilities':'The electric utility company/companies that serve the area where the vehicle is registered',
            'location':'city and county in which the vehicle is registered' }


    for col, meaning in cols.items():
        with st.sidebar.expander(col):
            st.write(meaning)




# visualization and analysis page

elif page == 'Analysis':

    avg_range = round(df['electric range'].mean(),2)
    total_brands = len(df['make'].unique())
    total_models = len(df['model'].unique())
    st.sidebar.metric('Average Range',avg_range)
    st.sidebar.metric('Total Brands',total_brands)
    st.sidebar.metric('Total Models',total_models)

    tab_1,tab_2,tab_3,tab_4,tab_5,tab_6,tab_7 = st.tabs(['electric range over time','vehicle type vs. electric range',
                          'electric range  vs. cafv eligibility','TOP 5 Brands','electric range distribution','cafv eligibility distribution',
                           'summary'])
    with tab_1:
        mean = df.groupby('model year')['electric range'].mean().reset_index()
        line = px.line(mean,y='electric range',x='model year',template='plotly_dark')
        st.plotly_chart(line,use_container_width=True)

    with tab_2:

        group = df.groupby('electric vehicle type')['electric range'].mean().reset_index()
        bar_1 = px.bar(group,y='electric vehicle type',x='electric range',barmode='group'
                       ,template='plotly_dark')
        st.plotly_chart(bar_1,use_container_width=True)


    with tab_3:
        group_2 = df.groupby('clean alternative fuel vehicle (cafv) eligibility')['electric range'].mean().reset_index()
        bar_2 = px.bar(group_2,y='electric range',x='clean alternative fuel vehicle (cafv) eligibility',barmode='group',template='plotly_dark')
        st.plotly_chart(bar_2,use_container_width=True)



    with tab_4:
        group_3 = df.groupby('make')['electric range'].mean().reset_index().sort_values(by='electric range',ascending=False)
        pie=px.pie(group_3.head(5),values='electric range',names='make',
                   color_discrete_sequence=px.colors.sequential.Blues_r,template='plotly_dark')
        st.plotly_chart(pie,use_container_width=True)

    with tab_5 :
        fig= px.histogram(df,x='electric range',nbins=5,
                     color_discrete_sequence=['slateblue'],template='plotly_dark')
        st.plotly_chart(fig,use_container_width = True)

    with tab_6 :

        fig_2 = px.histogram(df,x='clean alternative fuel vehicle (cafv) eligibility',
                     color='electric vehicle type',color_discrete_sequence= ['royalblue','blueviolet'],template='plotly_dark')
        st.plotly_chart(fig_2,use_container_width = True)            


    with tab_7:
        st.markdown(  
            'Battery Electric Vehicles tend to have higher average electric range , so it is highly recommended in manufacturing.'
               )
        st.markdown(
            """The electic range of electric cars manufactured in  Washington State is not stable over time but it has reached 
                its peak in 2010 and 2020. """)

        st.markdown('The eligibility for clean alternative fuel is not highly dependent on the vehicle type.')


# prediction page

else:

    st.image('https://media.istockphoto.com/id/2230080278/photo/electric-car-charging.jpg?s=2048x2048&w=is&k=20&c=nRUwCt8VrrhG-JwWyON2jTwM6nL522HOcKWLDgFaZLQ=')

    st.sidebar.header('Filters')

    # filters and inputs

    year = st.sidebar.selectbox('Year',df['model year'].unique())


    brand =st.sidebar.selectbox('Brand',df['make'].unique())

    # filter models based on selected brand
    filtered_models = df[df['make'] == brand]['model'].unique()
    car_model = st.sidebar.selectbox('Model', filtered_models)



    cafv =st.sidebar.radio('clean alternative',df['clean alternative fuel vehicle (cafv) eligibility'].unique())

    e_range = st.sidebar.slider('electric range',min_value=float(df['electric range'].min()),
                                                max_value= float(df['electric range'].max()))

    district =st.sidebar.selectbox('District No.',df['legislative district'].unique())

    utilities =st.sidebar.radio('number of utilities',df['number of utilities'].unique())

    # filtered location based on selected brand and model
    filtered_locations = df[(df['make'] == brand) & (df['model'] == car_model)]['location'].unique()
    location = st.sidebar.selectbox('Location', filtered_locations)



    # Prediction

    if st.button("Predict Vehicle Type"):

        new_df = pd.DataFrame(data= [[year,brand, car_model,cafv, e_range, district, utilities,location ]],
        columns=df.drop('electric vehicle type',axis=1).columns)

        st.table(new_df)

        result = model.predict(new_df)[0]

        if result == 0:
            st.write('Battery Electric Vehicle (BEV)')
        elif result == 1:
            st.write('Plug-in Hybrid Electric Vehicle (PHEV)')






