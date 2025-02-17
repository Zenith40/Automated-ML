# --------------------- Importing Libraries -------------------- 

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.impute import SimpleImputer
import pickle
import io

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score



# --------------------- Web-Page Design -------------------- 

st.set_page_config(
    page_title="Automate ML",
    layout="centered"
)

st.title('Automate ML Model')

uploaded_file = st.sidebar.file_uploader("Choose a file")


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)


    dataframe = st.sidebar.checkbox("Data Frame",True)
    describe = st.sidebar.checkbox("Distribution of Data Points",True)
    missing_per = st.sidebar.checkbox("Missing Value Percentage",True)

    num = [i for i in df.columns if df[i].dtype != 'object']

    if st.sidebar.button("Show Analysis", use_container_width=True):

        if dataframe:
            st.subheader('Data Frame : ')
            st.dataframe(df)
            st.write(' ')
            st.write(' ')
    

        if describe:
            st.subheader('Data Information : ')
            st.dataframe(df.describe())
            st.write(' ')
            st.write(' ')

        if missing_per:
            st.subheader('Missing Value Percentage')
            st.dataframe(df.isnull().sum()*100/df.shape[0])
            st.write(' ')
            st.write(' ')


    # Split
    label = st.sidebar.selectbox('Choose Y : ',df[num].columns.tolist(),index=len(df[num].columns.tolist())-1)

    x = df.drop(columns=[label],axis=1)   
    y = df[label]
    x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.25)



    # Data Processing
    scaler_dict = {"Min Max Scaling":MinMaxScaler(), "Standard Scaler" : StandardScaler()}

    scaler = st.sidebar.selectbox('Choose a Scaling Method : ',scaler_dict.keys())

    strategy = st.sidebar.selectbox('Choose Imputer : ',['mean', 'most_frequent', 'median'])

    impute = SimpleImputer(missing_values=np.nan,strategy=strategy)


    # Train
    model_dict = {
        "Logistic Regression": LogisticRegression(),
        "SVC" : SVC(),
        "Random Forest Classifier" : RandomForestClassifier(),
        "XGB Classifier" : XGBClassifier()
    }

    model = st.sidebar.selectbox('Choose the model : ', model_dict.keys())


    # Score    

    if st.sidebar.button("Process", use_container_width=True):
        st.subheader('Split Data Shape : ')
        st.write(f'x_train shape : {x_train.shape}')
        st.write(f'x_train shape : {y_train.shape}')
        st.write(f'x_train shape : {x_test.shape}')
        st.write(f'x_train shape : {y_test.shape}')

        st.write('')
        st.write('')

        st.subheader('Scaled Data : ')
        st.write(scaler_dict[scaler].fit_transform(x_train))

        st.write('')
        st.write('')

        x_train = impute.fit_transform(x_train)

        st.subheader(f'Cross-Validation Score : {np.around(cross_val_score(model_dict[model], x_train, y_train, cv=5).mean(),2)}')
        model_dict[model].fit(x_train,y_train)
        st.success("Model Trained Successfully")

        st.write('')
        st.write('')

        x_test = impute.transform(x_test)
        st.subheader(f'Prediction Accuracy on Validation Set : {np.around(accuracy_score(model_dict[model].predict(x_test),y_test),2)}')

    st.sidebar.subheader("Train the Model on Whole Dataset :")

    def pickle_model(model):
        f = io.BytesIO()
        pickle.dump(model, f)
        return f

    if st.sidebar.button("Train", use_container_width=True):

        scaler_dict[scaler].fit_transform(x)
        x = impute.fit_transform(x)
        st.subheader(f'Cross-Validation Score : {np.around(cross_val_score(model_dict[model], x, y, cv=5).mean(),2)}')
        model_dict[model].fit(x ,y)
        st.success("Model Trained Successfully")

        
        data = pickle_model(model)
        st.download_button("Download .pkl file", data=data.getvalue(), file_name="my-pickled-model.pkl", use_container_width=True)

    if st.sidebar.button("Future Scope", use_container_width=True):
        st.checkbox('Vizualization')
        st.checkbox('Regression, Clustering')
        st.checkbox('Handling Categorical Features')
        st.checkbox('Prediction')