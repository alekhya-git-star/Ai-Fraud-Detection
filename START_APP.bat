@echo off
title FraudShield AI
echo.
echo  Starting FraudShield AI...
echo.
cd /d "%~dp0"
pip install streamlit plotly pandas numpy scikit-learn joblib imbalanced-learn shap matplotlib -q
streamlit run app.py
pause
