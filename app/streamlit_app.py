import json
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, '/opt/airflow')
from src.config import MODEL, REPORTS

st.set_page_config(page_title='Mental health risk', layout='wide')
st.title('Student mental health risk')

if not MODEL.exists() or not (REPORTS / 'metrics.json').exists():
    st.error('Model not found. Run the Airflow DAG first.')
    st.stop()

page = st.sidebar.radio('Page', ['Model performance', 'Test the model'])

if page == 'Model performance':
    metrics = json.loads((REPORTS / 'metrics.json').read_text())

    cols = st.columns(5)
    for col, name in zip(cols, ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):
        col.metric(name, f'{metrics[name]:.3f}')

    st.bar_chart(pd.Series({k: metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}))
    st.image(str(REPORTS / 'confusion_matrix.png'), caption='Confusion matrix (test set)')

else:
    model = joblib.load(MODEL)

    with st.form('student'):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input('Age', 17, 29, 21)
        year = c1.number_input('Year of study', 1, 5, 2)
        gender = c1.selectbox('Gender', ['Female', 'Male', 'Non-binary'])
        sleep = c1.number_input('Sleep hours / day', 3.0, 10.0, 6.7)
        study = c1.number_input('Study hours / week', 0.0, 40.0, 20.0)
        gpa = c2.number_input('GPA', 1.5, 4.0, 2.56)
        social = c2.number_input('Social media hours / day', 0.0, 10.0, 6.0)
        screen = c2.number_input('Screen time hours / day', 0.0, 14.0, 8.5)
        attendance = c2.number_input('Attendance rate (%)', 50.0, 100.0, 85.0)
        stress = c3.number_input('Financial stress (1-10)', 1.0, 10.0, 6.0)
        support = c3.number_input('Family support (1-10)', 1.0, 10.0, 6.5)
        activity = c3.selectbox('Physical activity', ['Low', 'Moderate', 'High'])
        diet = c3.selectbox('Diet quality', ['Poor', 'Average', 'Good'])
        job = c3.selectbox('Part-time job', ['No', 'Yes'])
        history = c3.selectbox('Previous mental health history', ['No', 'Yes'])
        submitted = st.form_submit_button('Predict')

    if submitted:
        if social > screen:
            st.error('Social media time cannot be higher than total screen time.')
            st.stop()

        student = pd.DataFrame([{
            'Age': age, 'Gender': gender, 'Year_of_Study': year, 'Sleep_Hours': sleep,
            'Study_Hours_Per_Week': study, 'GPA': gpa, 'Social_Media_Hours': social,
            'Screen_Time_Hours': screen, 'Physical_Activity': activity,
            'Financial_Stress': stress, 'Family_Support': support,
            'Attendance_Rate': attendance, 'Part_Time_Job': job, 'Diet_Quality': diet,
            'Previous_Mental_Health_History': history,
        }])

        prediction = model.predict(student)[0]
        proba = pd.Series(model.predict_proba(student)[0], index=model.classes_)

        st.subheader(f'Predicted risk level: {prediction}')
        st.bar_chart(proba[['Low', 'Medium', 'High']])