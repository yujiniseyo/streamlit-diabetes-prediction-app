import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
# from lightgbm import LGBMClassifier
import os 
from sklearn.impute import SimpleImputer
# from imblearn.over_sampling import SMOTE


def run_ml_app():
    st.subheader('Machine Learning')

    menu = ['Xgboost' , 'Random Forest' , 'SVC' , 'XgBoost - After Over sampling' , 'Random Forest - After Over sampling' , 'SVC - After Over sampling']
    choice = st.selectbox('예측 모델을 선택하세요.', menu)

    new_data - np.array([3,88,58,11,54,24,0.26,22])
    new_data = new_data.reshape(1, -1)

    if choice == 'Xgboost' :
        st.write('Xgboost를 선택하셨습니다.')
        st.write('Xgboost는 여러개의 Decision Tree를 조합하여, Regression과 Classification 문제를 모두 지원하며,')
        st.write('성능과 자원 효율이 좋아서 인기 있게 사용되는 알고리즘 입니다.')

        model = joblib.load('data/xgboostbest_model.pkl')
        st.write(model.predict(new_data))



    elif choice == 'Random Forest' :
        st.write('Random Forest를 선택하셨습니다.')

        model = joblib.load('data/random_forestbest_model.pkl')
        st.write(model.predict(new_data))

    elif choice == 'SVC' :
        st.write('SVC를 선택하셨습니다.')

        model = joblib.load('data/svc_model.pkl')
        st.write(model.predict(new_data))

    elif choice == 'XgBoost - After Over sampling' :
        st.write('XgBoost - After Over sampling을 선택하셨습니다.')

        model = joblib.load('data/oversam_xgboostbest_model.pkl')
        st.write(model.predict(new_data))

    elif choice == 'Random Forest - After Over sampling' :
        st.write('Random Forest - After Over sampling을 선택하셨습니다.')

        model = joblib.load('data/oversam_random_forestbest_model.pkl')
        st.write(model.predict(new_data))

    elif choice == 'SVC - After Over sampling' :
        st.write('SVC - After Over sampling을 선택하셨습니다.')

        model = joblib.load('data/oversam_svcbest_model.pkl')
        st.write(model.predict(new_data))