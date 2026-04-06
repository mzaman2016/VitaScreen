@echo off
REM Run XAI RF Method - 8 sample heatmaps (TP, TN, FP, FN)
REM Place cdcNormalDiabeticFE1_20RFFSQ.csv in this folder or set path below

cd /d "%~dp0"
pip install -r requirements.txt -q
python rf_xai_method.py
pause
