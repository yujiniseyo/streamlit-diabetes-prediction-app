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
import joblib
import altair as alt 
from sklearn.impute import SimpleImputer

from eda_app import run_eda_app
from ml_app import run_ml_app
from about_app import run_about_app


st.set_page_config(page_title = 'Diabetes Prediction App made by yujiniseyo' , page_icon = '💚' , layout = 'wide' , initial_sidebar_state = 'collapsed')

def main() :
    st.title('당뇨병 예측 앱')

    menu = ['Home' , 'EDA' , 'Predict' , 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home' :
        st.image('https://miro.medium.com/max/2625/1*INSggrGiQ1lCgU8YTsfEVw.png')
        st.write('이 앱은 환자 데이터를 바탕으로 당뇨병을 예측하는 앱입니다.')
        st.write("이 앱에 사용된 인공지능 예측 모델은 'over sampling - random forest'이고, 정확도는 83.5% 입니다 !" )
        st.write('왼쪽의 사이드바에서 원하시는 메뉴를 선택하세요.')

    elif choice == 'EDA' :
        st.image('https://miro.medium.com/max/2625/1*INSggrGiQ1lCgU8YTsfEVw.png')
        run_eda_app()

    elif choice == 'Predict' :
        st.image('https://miro.medium.com/max/2625/1*INSggrGiQ1lCgU8YTsfEVw.png')
        run_ml_app()

    elif choice == 'About' :
        run_about_app()







if __name__ == '__main__' :
    main()