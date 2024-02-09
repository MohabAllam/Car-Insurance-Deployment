
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from imblearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from category_encoders import BinaryEncoder

select_page = st.sidebar.radio('Select page', ['Analysis', 'Model Classification'])
if select_page == 'Analysis':
    
    def main():
        cleaned_df = pd.read_csv('cleaned_df.csv')
        st.image('EDA.jpg')
        st.write('### Head of Dataframe')
        st.dataframe(cleaned_df.head(10))
        tab1, tab2, tab3 = st.tabs(['Univariate Analysis', 'Bivariate Analysis', 'Multivariate Analysis'])
        
        # Univariate Analysis
        tab1.write('### Univariate Analysis with Histogram for each Feature')
        for col in cleaned_df.columns:
            tab1.plotly_chart(px.histogram(cleaned_df, x= col))
         
        # Bivariate: Plot Numerical vs Target feature
        
        tab2.write('### Numerical Features vs Target Variable')
        
        select_plot = tab2.selectbox('Select PLot Type', ['Boxplot', 'Violinplot', 'Stripplot'])

        select_feature = tab2.selectbox('Select Feature', ['age', 'homekids', 'yoj', 'income', 'home_val', 'travtime', 'bluebook', 'tif', 'oldclaim', 'clm_freq', 'mvr_pts', 'clm_amt', 'car_age'])

        if select_plot == 'Boxplot':
            tab2.plotly_chart(px.box(cleaned_df, 'claim_flag', select_feature))

        elif select_plot == 'Violinplot':
            tab2.plotly_chart(px.violin(cleaned_df, 'claim_flag', select_feature))

        else:
            tab2.plotly_chart(px.strip(cleaned_df, 'claim_flag', select_feature))
        
        # Multivaariate
        tab3.write('### Hearmap for Numerical Features')
        
        num_cols = cleaned_df.select_dtypes(exclude= 'O').columns.to_list()[:-1]
        num_cols_df = cleaned_df.select_dtypes(exclude= 'O').drop('claim_flag', axis= 1)
        tab3.plotly_chart(px.imshow(num_cols_df.corr(), x= num_cols))
        
    if __name__=='__main__':
        main()  
        
        
        
        
        
elif select_page == 'Model Classification':
    
    def main():
        st.title('Model Classification')
        st.image('https://images.moneycontrol.com/static-mcnews/2018/03/car-insurance-770x433.jpg?impolicy=website&width=770&height=431')
        # First step: load pkl file
        pipeline = joblib.load('RF_pipeline.pkl')

        # Second step: Create dataframe from input data
        def Prediction(kidsdriv, age, homekids, yoj, income, parent1, home_val,
               mstatus, gender, education, occupation, travtime, car_use,
               bluebook, tif, car_type, red_car, oldclaim, clm_freq,
               revoked, mvr_pts, clm_amt, car_age):

            df = pd.DataFrame(columns= ['kidsdriv', 'age', 'homekids', 'yoj', 'income', 'parent1', 'home_val','mstatus', 'gender', 'education', 'occupation', 'travtime', 'car_use','bluebook', 'tif', 'car_type', 'red_car', 'oldclaim', 'clm_freq','revoked', 'mvr_pts', 'clm_amt', 'car_age'])

            df.at[0, 'kidsdriv'] = kidsdriv
            df.at[0, 'age'] = age
            df.at[0, 'homekids'] = homekids
            df.at[0, 'yoj'] = yoj
            df.at[0, 'income'] = income
            df.at[0, 'parent1'] = parent1
            df.at[0, 'home_val'] = home_val
            df.at[0, 'mstatus'] = mstatus
            df.at[0, 'gender'] = gender
            df.at[0, 'education'] = education
            df.at[0, 'occupation'] = occupation
            df.at[0, 'travtime'] = travtime
            df.at[0, 'car_use'] = car_use
            df.at[0, 'bluebook'] = bluebook
            df.at[0, 'tif'] = tif
            df.at[0, 'car_type'] = car_type
            df.at[0, 'red_car'] = red_car
            df.at[0, 'oldclaim'] = oldclaim
            df.at[0, 'clm_freq'] = clm_freq
            df.at[0, 'revoked'] = revoked
            df.at[0, 'mvr_pts'] = mvr_pts
            df.at[0, 'clm_amt'] = clm_amt
            df.at[0, 'car_age'] = car_age

            # Make prediction of dataframe using pipeline
            result = pipeline.predict(df)[0]
            return result


        kidsdriv = st.selectbox('PLease provid number of driving kids', [0,1,2,3,4])
        age = st.sidebar.slider('Enter your Age', 16, 81)
        homekids = st.selectbox('PLease provid total number of kids', [0,1,2,3,4, 5])
        yoj = st.sidebar.slider('Please provide number of years on job', 0, 23)
        income = st.sidebar.slider('Please provide your income')
        parent1 = st.selectbox('PLease select whether you are single parent or not', ['Yes', 'No'])
        home_val = st.sidebar.slider('Please provide your home value')
        mstatus = st.sidebar.radio('Marital Status', ['Yes', 'No'])
        gender = st.sidebar.radio('Gender', ['M', 'F'])
        education = st.selectbox('Please enter your educational background', ['PhD', 'High School', 'Bachelors', '<High School', 'Masters'])
        occupation = st.selectbox('Please enter your Occupation', ['Professional', 'Blue Collar', 'Manager', 'Clerical', 'Doctor', 'Lawyer','Home Maker', 'Student'])
        travtime = st.sidebar.slider('Please enter your travel time in minutes', 5, 142)
        car_use = st.selectbox('PLease provid your car usage', ['Private', 'Commercial'])
        bluebook = st.sidebar.slider('Please provide your car value')
        tif = st.sidebar.slider('Loyalty years', 1, 25)
        car_type = st.selectbox('PLease provid your car type', ['Minivan', 'Van', 'SUV', 'Sports Car', 'Panel Truck', 'Pickup'])
        red_car = st.selectbox('Red car or Not', ['yes', 'no'])
        oldclaim = st.sidebar.slider('Please provide your old claim amount')
        clm_freq = st.selectbox('Please provide number of previous claims', [0, 1, 2, 3, 4, 5])
        revoked = st.selectbox('License Revoked within 7 years', ['No', 'Yes'])
        mvr_pts = st.sidebar.slider('Please provide vechile record points', 0, 13)
        clm_amt = st.sidebar.slider('Please provide your total claims amount')
        car_age = st.sidebar.slider('Please enter your car age', 0, 28)

        if st.button('Predict'):
            result = Prediction(kidsdriv, age, homekids, yoj, income, parent1, home_val,
                                mstatus, gender, education, occupation, travtime, car_use,
                                bluebook, tif, car_type, red_car, oldclaim, clm_freq,
                                revoked, mvr_pts, clm_amt, car_age)
            if result == 1:
                st.write('This customer will probably make a claim')
            elif result == 0:
                st.write('This cust will not probably make a claim')
        
    if __name__=='__main__':
        main()    
    
