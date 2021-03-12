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

from eda_app import run_eda_app
from ml_app import run_ml_app


def main() :
    st.title('당뇨병 예측 앱')

    menu = ['Home' , 'EDA' , 'ML']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home' :
        st.write('이 앱은 환자 정보를 바탕으로 당뇨병을 예측하는 앱입니다.')
        st.write('왼쪽의 사이드바에서 선택하세요.')

    elif choice == 'EDA' :
        run_eda_app()

    elif choice == 'ML' :
        run_ml_app()






if __name__ == '__main__' :
    main()