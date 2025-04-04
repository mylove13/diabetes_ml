
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 앱 타이틀
st.title("🩺 당뇨병 위험 예측기")

# 데이터 로딩 함수
def load_data():
    return pd.read_csv("diabetes.csv")

# 데이터 전처리 함수
def preprocess_data(df):
    df = df.copy()
    # 0으로 입력된 이상치 대체 (해당 컬럼들만)
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[col] = df[col].replace(0, np.nan)
        df[col].fillna(df[col].mean(), inplace=True)

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, y, scaler, X.columns

# 모델 학습 함수
def train_model(X_train, y_train, max_depth, n_estimators):
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model

# 사이드바 - 사용자 입력
st.sidebar.header("🧪 사용자 건강 정보 입력")
pregnancies = st.sidebar.slider("임신 횟수", 0, 20, 1)
glucose = st.sidebar.slider("포도당 농도", 50, 200, 120)
blood_pressure = st.sidebar.slider("이완기 혈압", 30, 130, 70)
skin_thickness = st.sidebar.slider("피부 두께", 0, 100, 20)
insulin = st.sidebar.slider("인슐린 수치", 0, 900, 80)
bmi = st.sidebar.slider("BMI", 10.0, 70.0, 25.0)
diabetes_pedigree = st.sidebar.slider("가족력 지수", 0.0, 2.5, 0.5)
age = st.sidebar.slider("나이", 18, 100, 33)

user_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [diabetes_pedigree],
    'Age': [age]
})

# 하이퍼파라미터 입력
st.sidebar.header("⚙️ 모델 설정")
max_depth = st.sidebar.slider("max_depth", 1, 20, 5)
n_estimators = st.sidebar.slider("n_estimators", 10, 300, 100, step=10)

# 실행 버튼
if st.button("🚀 예측 실행"):
    df = load_data()
    X, y, scaler, feature_names = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train, max_depth, n_estimators)

    # 유저 입력 스케일 적용
    user_scaled = scaler.transform(user_data)
    prediction = model.predict(user_scaled)[0]
    pred_proba = model.predict_proba(user_scaled)[0][1]

    # 예측 결과 출력
    st.subheader("🔍 예측 결과")
    if prediction == 1:
        st.error(f"당뇨병 위험이 있습니다. (확률: {pred_proba:.2%})")
    else:
        st.success(f"당뇨병 위험이 낮습니다. (확률: {pred_proba:.2%})")

    # 모델 성능 출력
    st.subheader("📊 모델 성능 평가")
    y_pred = model.predict(X_test)
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    st.pyplot(fig)

    # 📈 피처 중요도 시각화
    st.subheader("🔬 특성 중요도 (Feature Importance)")
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
    ax_imp.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    ax_imp.set_xlabel("Importance")
    ax_imp.set_title("Feature Importance")
    st.pyplot(fig_imp)

    # 저장 버튼    
    #if st.button("💾 모델 저장"):
    #    model_filename = "diabetes_model.pkl"
    #    joblib.dump(model, model_filename)
    #    st.success(f"모델이 {model_filename}로 저장되었습니다.")
