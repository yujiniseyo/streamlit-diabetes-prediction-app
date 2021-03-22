import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
from lightgbm import LGBMClassifier
import os 
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib
import altair as alt 
from sklearn.impute import SimpleImputer


def run_eda_app() :
    st.subheader('Exploratory Data Analysis, 탐색적 데이터 분석')

    data = 'data/diabetes.csv'

    df = pd.read_csv(data)
    
   
    radio_menu = ['Date Frame' , 'Describe']
    selected_radio = st.radio('데이터 프레임 / 분석 정보', radio_menu)

    if selected_radio == 'Date Frame' :
        st.dataframe(df)

    elif selected_radio == 'Describe' :
        st.dataframe(df.describe())

    columns = df.columns
    columns = list(columns)

    selected_cols = st.multiselect('원하시는 데이터 컬럼을 선택하세요.' , columns)
    if len(selected_cols) != 0 :
        st.dataframe(df[selected_cols])
    else :
        st.write('선택한 컬럼이 없습니다.')


    # 상관계수를 화면에 보여주도록 만든다.
    # 멀티셀렉트에 컬럼명 보여주고, 해당 컬럼들에 대한 상관계수를 보여준다.

    corr_cols = df.columns
    selected_corr = st.multiselect('원하시는 상관계수 컬럼을 선택하세요.' , corr_cols)

    if len(selected_corr) != 0 :
        st.dataframe(df[selected_corr].corr())
        
        # 위에서 선택한 컬럼들을 이용해서, sns.pairplot을 그린다.

        if 'Outcome' in selected_corr :
            fig = sns.pairplot(data = df[selected_corr], hue = 'Outcome')
            st.pyplot(fig)

        else :
            fig = sns.pairplot(data = df[selected_corr])
            st.pyplot(fig)

    else :
        st.write('선택한 컬럼이 없습니다.')


    
    graph_menu = ['Choose an option' , 'Heat Map' , 'Histogram' , 'Pie Chart']
    selected_graph = st.selectbox('원하시는 차트를 선택하세요.', graph_menu)

    if selected_graph == 'Heat Map' :
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure(figsize = (20,20))
        sns.heatmap(df.corr(), annot = True, vmax = 1, vmin = -1)
        st.pyplot()

    elif selected_graph == 'Histogram' :
        st.set_option('deprecation.showPyplotGlobalUse', False)
        df.hist(figsize = (20,20))
        st.pyplot()
    
    elif selected_graph == 'Pie Chart' :
        pass

    elif selected_graph == 'Choose an option' :
        st.write('선택한 차트가 없습니다.')
    


    
    
    
    
    
    # # 히트맵 보기 버튼을 누르면 히트맵을 보여준다.

    # heatmap_bnt = st.button('Show Heatmap')
    # if heatmap_bnt :
    #     st.set_option('deprecation.showPyplotGlobalUse', False)
    #     sns.heatmap(df.corr(), annot = True, vmax = 1, vmin = -1)
    #     st.pyplot()

    # # 히스토그램 보기 버튼을 누르면 히스토그램을 보여준다.

    # hist_bnt = st.button('Show Histogram')
    # if hist_bnt :
    #     st.set_option('deprecation.showPyplotGlobalUse', False)
    #     df.hist(figsize = (20,20))
    #     st.pyplot()