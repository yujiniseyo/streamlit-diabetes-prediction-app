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


def run_ml_app():
    st.subheader("Let's predict diabetes.")

    # input_data = []



    # new_data = np.array([6,148,72,35,54,33,0.62,50])
    # new_data = new_data.reshape(1, -1)

    Pregnancies = st.number_input('임신 횟수를 입력하세요.', min_value = 0)
    Glucose = st.number_input('공복 혈당 수치를 입력하세요.', min_value = 1)
    BloodPressure = st.number_input('혈압을 입력하세요.', min_value = 1)
    SkinThickness = st.number_input('피부 두께를 입력하세요.', min_value = 1)
    Insulin = st.number_input('인슐린 수치를 입력하세요.', min_value = 1)
    BMI = st.number_input('비만도를 입력하세요.', min_value = 1)
    Diabetes_pedigree_function = st.number_input('입력하세요.', min_value = 1)
    Age = st.number_input('나이를 입력하세요.', min_value = 1, max_value = 120)

    new_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Diabetes_pedigree_function, Age])
    new_data = new_data.reshape(1, -1)
  
    if st.button('Predict') :
        st.write('입력하신 정보로 당뇨병 예측을 시작합니다.')
        st.write('0 : 당뇨병 X.')
        st.write('1 : 당뇨병 O.')

        model = joblib.load('data/oversam_random_forestbest_model.pkl')
        result = st.text(model.predict(new_data))
        

