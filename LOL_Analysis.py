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


# 1. 승패 예측 모델 만들기

#협곡 데이터
df_classic = df[df['game_mode'] == 'CLASSIC']

df_drop = df_classic.drop(['game_start_utc','game_type','game_version', 'map_id', 'platform_id',
                           'participant_id','puuid', 'summoner_name', 'summoner_id', 'summoner_level',
                           'champion_id', 'champion_name', 'individual_position', 'lane', 'role',
                           'item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6',
                           'champion_mastery_lastPlayTime', 'champion_mastery_lastPlayTime_utc', 'champion_mastery_pointsSinceLastLevel','champion_mastery_pointsUntilNextLevel',
                           'champion_mastery_tokensEarned','final_abilityHaste','flex_tier', 'flex_rank', 'flex_lp',
       'flex_wins', 'flex_losses','game_mode',], axis = 'columns')

df_drop = df_drop.dropna(subset=['team_position'])

# 솔로랭크 티어의 결측치는 Unranked I 0lp 0승 0패
df_drop.loc[:, 'solo_tier'] = df_drop['solo_tier'].fillna("Unranked")
df_drop.loc[:, 'solo_rank'] = df_drop['solo_rank'].fillna("I")
df_drop.loc[:, 'solo_lp'] = df_drop['solo_lp'].fillna(0)
df_drop.loc[:, 'solo_wins'] = df_drop['solo_wins'].fillna(0)
df_drop.loc[:, 'solo_losses'] = df_drop['solo_losses'].fillna(0)

# 챔피언별 숙련도는 평균치로 결측치 보강
df_drop.loc[:, 'champion_mastery_level'] = df_drop['champion_mastery_level'].fillna(df_drop['champion_mastery_level'].mean())
df_drop.loc[:, 'champion_mastery_points'] = df_drop['champion_mastery_points'].fillna(df_drop['champion_mastery_points'].mean())

# 승패예측용 데이터 전처리

df_drop['solo_rank_tier'] = df_drop['solo_tier'] + '_' + df_drop['solo_rank']
df_drop = df_drop.drop(['solo_tier','solo_rank'], axis = 'columns')
df_drop = df_drop.drop(['queue_id'], axis = 'columns')

missing_values = df_drop.isnull().sum()
print('Missing values per column:')
print(missing_values[missing_values > 0])

# 1. 인코딩할 문자형 컬럼만 지정
categorical_cols = ['team_position', 'solo_rank_tier']

# 2. 숫자형 컬럼만 자동 추출 (object, bool 제외)
numerical_cols = df_drop.select_dtypes(exclude=['object', 'bool']).columns.tolist()
if 'win' not in numerical_cols and 'win' in df_drop.columns:
    numerical_cols.append('win')
    
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

#승패 예측 모델 함수
def evaluate_model_win(model, model_name, X_train, X_test, y_train, y_test, features, scale=True):
    # 👉 스케일러 포함한 파이프라인 생성
    if scale:
        model = make_pipeline(StandardScaler(), model)

    # 학습 및 예측
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'\n🔹 {model_name} - Prediction Accuracy: {acc:.4f}')
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred, digits=4))


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
    else:
        print(f"{model_name} does not support predict_proba, skipping ROC curve.")
    
    # Permutation Importance
    from sklearn.pipeline import Pipeline

    if isinstance(model, Pipeline):
        estimator = model.named_steps[list(model.named_steps.keys())[-1]]
    else:
        estimator = model
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
  
# 승패 예측 모델 실행
models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
    'Decision Tree': (DecisionTreeClassifier(random_state=42), True),
    'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42), False),
    'XGBoost': (XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42), False),
    'LightGBM': (LGBMClassifier(random_state=42), False)
}

for name, (model, use_scale) in models.items():
    evaluate_model_win(model, name, X_train, X_test, y_train, y_test, features, scale=use_scale)


# 2. 티어 예측 모델 만들기

# 티어 에측 모델 함수
def evaluate_model_tier(model, model_name, X_train, X_test, y_train, y_test, scale=True,label_encoder=None):
    features = X_train.columns 
    # 👉 스케일러 포함한 파이프라인 생성
    if scale:
        model = make_pipeline(StandardScaler(), model)

    # 학습 및 예측
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)    # ✅ 디코딩 (선택적으로)
    
    if label_encoder is not None:
        y_test_decoded = label_encoder.inverse_transform(y_test)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
    else:
        y_test_decoded = y_test
        y_pred_decoded = y_pred
    acc = accuracy_score(y_test, y_pred)
    
    print(f'\n🔹 {model_name} - Prediction Accuracy: {acc:.4f}')
    from sklearn.metrics import classification_report
    print(classification_report(y_test_decoded, y_pred_decoded, digits=4))


    # Confusion Matrix
    cm = confusion_matrix(y_test_decoded, y_pred_decoded)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    # ROC Curve
    if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
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
    else:
        print(f"{model_name} - ROC Curve 생략 (이진 분류 전용)")
    
    # Permutation Importance
    from sklearn.pipeline import Pipeline

    if isinstance(model, Pipeline):
        estimator = model.named_steps[list(model.named_steps.keys())[-1]]
    else:
        estimator = model
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

#협곡 데이터
df_classic = df[df['game_mode'] == 'CLASSIC']

df_drop = df_classic.drop(['game_start_utc','game_type','game_version', 'map_id', 'platform_id',
                           'participant_id','puuid', 'summoner_name', 'summoner_id', 'summoner_level',
                           'champion_id', 'champion_name', 'individual_position', 'lane', 'role',
                           'item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6',
                           'champion_mastery_lastPlayTime', 'champion_mastery_lastPlayTime_utc', 'champion_mastery_pointsSinceLastLevel','champion_mastery_pointsUntilNextLevel',
                           'champion_mastery_tokensEarned','final_abilityHaste','flex_tier', 'flex_rank', 'flex_lp',
       'flex_wins', 'flex_losses','game_mode',], axis = 'columns')

df_drop = df_drop.dropna(subset=['team_position'])

# 솔로랭크 티어의 결측치는 Unranked I 0lp 0승 0패
df_drop.loc[:, 'solo_tier'] = df_drop['solo_tier'].fillna("Unranked")
df_drop.loc[:, 'solo_rank'] = df_drop['solo_rank'].fillna("I")
df_drop.loc[:, 'solo_lp'] = df_drop['solo_lp'].fillna(0)
df_drop.loc[:, 'solo_wins'] = df_drop['solo_wins'].fillna(0)
df_drop.loc[:, 'solo_losses'] = df_drop['solo_losses'].fillna(0)

# 챔피언별 숙련도는 평균치로 결측치 보강
df_drop.loc[:, 'champion_mastery_level'] = df_drop['champion_mastery_level'].fillna(df_drop['champion_mastery_level'].mean())
df_drop.loc[:, 'champion_mastery_points'] = df_drop['champion_mastery_points'].fillna(df_drop['champion_mastery_points'].mean())

# 티어 예측용 데이터 전처리

# 'solo_tier' 값을 기반으로 그룹화된 티어 매핑
tier_mapping = {
    'Unranked': 'Unranked',
    'IRON': 'Iron-Bronze-Silver-Gold',
    'BRONZE': 'Iron-Bronze-Silver-Gold',
    'SILVER': 'Iron-Bronze-Silver-Gold',
    'GOLD': 'Iron-Bronze-Silver-Gold',
    'PLATINUM': 'Platinum-Emerald',
    'EMERALD': 'Platinum-Emerald',
    'DIAMOND': 'Diamond-Master',
    'MASTER': 'Diamond-Master',
    'GRANDMASTER': 'Grandmaster-Challenger',
    'CHALLENGER': 'Grandmaster-Challenger'
}


df_drop['solo_rank_group'] = df_drop['solo_tier'].map(tier_mapping)

df_drop = df_drop.drop(['solo_tier','solo_rank', 'queue_id','solo_lp','win','champion_mastery_level', 'champion_mastery_points','game_id'], axis='columns')

# 5. 인코딩할 컬럼 (타겟 제외!)
categorical_cols = ['team_position']  # 'solo_rank_tier' 제외!

# 6. 숫자형 컬럼 추출
numerical_cols = df_drop.select_dtypes(exclude=['object', 'bool']).columns.tolist()

# 7. 인코딩
df_encoded = pd.get_dummies(df_drop[categorical_cols], drop_first=False, dtype=int)

# 8. 최종 데이터 결합 (타겟 제외!)
df_final = pd.concat([df_drop[numerical_cols], df_encoded], axis=1)

# 9. 피처/타겟 분리
X = df_final
y = df_drop['solo_rank_group']

# 10. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import LabelEncoder

# 문자열 타겟 라벨 -> 숫자 인코딩
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# 모델 실행

models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
    'Decision Tree': (DecisionTreeClassifier(random_state=42), True),
    'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42), False),
    'XGBoost': (XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42), False),
    'LightGBM': (LGBMClassifier(random_state=42), False)
}

for name, (model, use_scale) in models.items():
    evaluate_model_tier(model, name, X_train, X_test, y_train, y_test, scale=use_scale,label_encoder=le)

# 위 모델은 solo_rank의 전체 승/패가 너무 큰 영향을 주게 됨. => solo_wins, solo_losses 항목을 제외하고 다시 학습.

# 승패 예측 모델 2 (같은 함수 사용, 데이터 전처리 과정만 변경)

#협곡 데이터
df_classic = df[df['game_mode'] == 'CLASSIC']

df_drop = df_classic.drop(['game_start_utc','game_type','game_version', 'map_id', 'platform_id',
                           'participant_id','puuid', 'summoner_name', 'summoner_id', 'summoner_level',
                           'champion_id', 'champion_name', 'individual_position', 'lane', 'role',
                           'item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6',
                           'champion_mastery_lastPlayTime', 'champion_mastery_lastPlayTime_utc', 'champion_mastery_pointsSinceLastLevel','champion_mastery_pointsUntilNextLevel',
                           'champion_mastery_tokensEarned','final_abilityHaste','flex_tier', 'flex_rank', 'flex_lp',
       'flex_wins', 'flex_losses','game_mode',], axis = 'columns')

df_drop = df_drop.dropna(subset=['team_position'])

# 솔로랭크 티어의 결측치는 Unranked I 0lp 0승 0패
df_drop.loc[:, 'solo_tier'] = df_drop['solo_tier'].fillna("Unranked")
df_drop.loc[:, 'solo_rank'] = df_drop['solo_rank'].fillna("I")
df_drop.loc[:, 'solo_lp'] = df_drop['solo_lp'].fillna(0)
df_drop.loc[:, 'solo_wins'] = df_drop['solo_wins'].fillna(0)
df_drop.loc[:, 'solo_losses'] = df_drop['solo_losses'].fillna(0)

# 챔피언별 숙련도는 평균치로 결측치 보강
df_drop.loc[:, 'champion_mastery_level'] = df_drop['champion_mastery_level'].fillna(df_drop['champion_mastery_level'].mean())
df_drop.loc[:, 'champion_mastery_points'] = df_drop['champion_mastery_points'].fillna(df_drop['champion_mastery_points'].mean())

# 티어 예측용 데이터 전처리

# 'solo_tier' 값을 기반으로 그룹화된 티어 매핑
tier_mapping = {
    'Unranked': 'Unranked',
    'IRON': 'Iron-Bronze-Silver-Gold',
    'BRONZE': 'Iron-Bronze-Silver-Gold',
    'SILVER': 'Iron-Bronze-Silver-Gold',
    'GOLD': 'Iron-Bronze-Silver-Gold',
    'PLATINUM': 'Platinum-Emerald',
    'EMERALD': 'Platinum-Emerald',
    'DIAMOND': 'Diamond-Master',
    'MASTER': 'Diamond-Master',
    'GRANDMASTER': 'Grandmaster-Challenger',
    'CHALLENGER': 'Grandmaster-Challenger'
}


df_drop['solo_rank_group'] = df_drop['solo_tier'].map(tier_mapping)

df_drop = df_drop.drop(['solo_tier','solo_rank', 'queue_id','solo_lp','win','champion_mastery_level', 'champion_mastery_points','game_id','solo_wins', 'solo_losses'], axis='columns')

# 5. 인코딩할 컬럼 (타겟 제외!)
categorical_cols = ['team_position']  # 'solo_rank_tier' 제외!

# 6. 숫자형 컬럼 추출
numerical_cols = df_drop.select_dtypes(exclude=['object', 'bool']).columns.tolist()

# 7. 인코딩
df_encoded = pd.get_dummies(df_drop[categorical_cols], drop_first=False, dtype=int)

# 8. 최종 데이터 결합 (타겟 제외!)
df_final = pd.concat([df_drop[numerical_cols], df_encoded], axis=1)

# 9. 피처/타겟 분리
X = df_final
y = df_drop['solo_rank_group']

# 10. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import LabelEncoder

# 문자열 타겟 라벨 -> 숫자 인코딩
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# 모델 실행

models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
    'Decision Tree': (DecisionTreeClassifier(random_state=42), True),
    'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42), False),
    'XGBoost': (XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42), False),
    'LightGBM': (LGBMClassifier(random_state=42), False)
}

for name, (model, use_scale) in models.items():
    evaluate_model_tier(model, name, X_train, X_test, y_train, y_test, scale=use_scale,label_encoder=le)
