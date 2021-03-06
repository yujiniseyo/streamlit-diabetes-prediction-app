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
# from lightgbm import LGBMClassifier
import os 
from sklearn.impute import SimpleImputer
# from imblearn.over_sampling import SMOTE
import joblib
import altair as alt 
from sklearn.impute import SimpleImputer

from eda_app import run_eda_app
from ml_app import run_ml_app
from about_app import run_about_app


st.set_page_config(page_title = 'Diabetes Prediction App made by yujiniseyo' , page_icon = 'π' , layout = 'wide' , initial_sidebar_state = 'collapsed')

def main() :
    st.title('λΉλ¨λ³ μμΈ‘ μ±')

    st.image('https://miro.medium.com/max/2625/1*INSggrGiQ1lCgU8YTsfEVw.png')

    menu = ['Home' , 'EDA' , 'Predict' , 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home' :
        st.write('μ΄ μ±μ νμ λ°μ΄ν°λ₯Ό λ°νμΌλ‘ λΉλ¨λ³μ μμΈ‘νλ μ±μλλ€.')
        st.write("μ΄ μ±μ μ¬μ©λ μΈκ³΅μ§λ₯ μμΈ‘ λͺ¨λΈμ 'over sampling - random forest'μ΄κ³ , μ νλλ 83.5% μλλ€ !" )
        st.write('μΌμͺ½μ μ¬μ΄λλ°μμ μνμλ λ©λ΄λ₯Ό μ ννμΈμ.')

    elif choice == 'EDA' :
        run_eda_app()

    elif choice == 'Predict' :
        run_ml_app()

    elif choice == 'About' :
        run_about_app()







if __name__ == '__main__' :
    main()