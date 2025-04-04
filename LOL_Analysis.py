import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')  # Important for rendering figures
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Switching backend if needed
%matplotlib inline

import seaborn as sns
sns.set(style='whitegrid', palette='muted', color_codes=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


print('Libraries imported successfully')

# Load the dataset from the Excel file
data_path = './archive/league_data.xlsx'

# Specify that 'game_start_utc' and 'champion_mastery_lastPlayTime_utc' should be parsed as dates
date_cols = ['game_start_utc', 'champion_mastery_lastPlayTime_utc']
df = pd.read_excel(data_path, parse_dates=date_cols, engine='openpyxl')

print('Data loaded successfully')
print('Dataset shape:', df.shape)

#협곡 데이터
df_classic = df[df['game_mode'] == 'CLASSIC']

df_drop = df_classic.drop(['game_start_utc','game_type','game_version', 'map_id', 'platform_id',
                           'participant_id','puuid', 'summoner_name', 'summoner_id', 'summoner_level',
                           'champion_id', 'champion_name', 'individual_position', 'lane', 'role',
                           'item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6',
                           'champion_mastery_lastPlayTime', 'champion_mastery_lastPlayTime_utc', 'champion_mastery_pointsSinceLastLevel','champion_mastery_pointsUntilNextLevel',
                           'champion_mastery_tokensEarned','final_abilityHaste'], axis = 'columns')

# 자랭 데이터 삭제
df_drop = df_drop.drop(['flex_tier', 'flex_rank', 'flex_lp',
       'flex_wins', 'flex_losses'], axis = 'columns')

df_drop = df_drop.drop('game_mode', axis = 'columns')
df_drop = df_drop.drop(['solo_lp'], axis = 'columns')

df_drop = df_drop.dropna(subset=['team_position'])

df_drop.loc[:, 'solo_tier'] = df_drop['solo_tier'].fillna("Unranked")
df_drop.loc[:, 'solo_rank'] = df_drop['solo_rank'].fillna("I")
df_drop.loc[:, 'solo_lp'] = df_drop['solo_lp'].fillna(0)
df_drop.loc[:, 'solo_wins'] = df_drop['solo_wins'].fillna(0)
df_drop.loc[:, 'solo_losses'] = df_drop['solo_losses'].fillna(0)

df_drop.loc[:, 'champion_mastery_level'] = df_drop['champion_mastery_level'].fillna(df_drop['champion_mastery_level'].mean())
df_drop.loc[:, 'champion_mastery_points'] = df_drop['champion_mastery_points'].fillna(df_drop['champion_mastery_points'].mean())

df_drop['solo_rank_tier'] = df_drop['solo_tier'] + ' ' + df_drop['solo_rank']
df_drop = df_drop.drop(['solo_tier','solo_rank'], axis = 'columns')
df_drop = df_drop.drop(['queue_id'], axis = 'columns')

missing_values = df_drop.isnull().sum()
print('Missing values per column:')
print(missing_values[missing_values > 0])

# 1. 인코딩할 문자형 컬럼만 지정
categorical_cols = ['team_position', 'solo_rank_tier']

# 2. 숫자형 컬럼만 자동 추출 (object, bool 제외)
numerical_cols = df_drop.select_dtypes(exclude=['object', 'bool']).columns.tolist()

# 3. 원-핫 인코딩 적용 (선택한 문자형 컬럼만)
df_encoded = pd.get_dummies(df_drop[categorical_cols], drop_first=False, dtype=int)  # drop_first=True 옵션도 가능

# 4. 숫자형 컬럼과 인코딩된 컬럼 합치기
df_final = pd.concat([df_drop[numerical_cols], df_encoded], axis=1)

drop_win = df_final.drop('win', axis = 'columns')
features = drop_win.columns

# Ensure our target is numeric (boolean True/False usually work fine but we convert to int for scoring)
df_final['win'] = df_final['win'].astype(int)
target = 'win'

X = df_final[features]
y = df_final[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model, model_name, scale=True):
    # 👉 스케일러 포함한 파이프라인 생성
    if scale:
        model = make_pipeline(StandardScaler(), model)

    # 학습 및 예측
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'\n🔹 {model_name} - Prediction Accuracy: {acc:.4f}')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    # ROC Curve
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

    # Permutation Importance
    r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    top_idx = np.argsort(r.importances_mean)[-15:][::-1]
    top_features = [features[i] for i in top_idx]
    top_importances = r.importances_mean[top_idx]
    top_errors = r.importances_std[top_idx]

    plt.figure(figsize=(8, 6))
    plt.barh(top_features, top_importances, xerr=top_errors, color='teal')
    plt.xlabel('Mean Importance')
    plt.title(f'{model_name} - Top 15 Permutation Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# ---------- 모델 실행 ----------

# 1. Logistic Regression

evaluate_model(LogisticRegression(max_iter=1000,random_state=42), 'Logistic Regression')

# 2. Decision Tree

evaluate_model(DecisionTreeClassifier(random_state=42), 'Decision Tree')

# 3. Random Forest
evaluate_model(RandomForestClassifier(n_estimators=100,random_state=42), 'Random Forest', scale=False)

# 4. XGBoost
evaluate_model(XGBClassifier(n_estimators=100, eval_metric='logloss',random_state=42), 'XGBoost', scale=False)

# 5. LightGBM
evaluate_model(LGBMClassifier(random_state=42), 'LightGBM', scale=False)
