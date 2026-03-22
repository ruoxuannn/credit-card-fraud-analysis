import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

print("Loading dataset...")
df = pd.read_csv('creditcard.csv')

print(f"Shape: {df.shape}")
print(f"Fraud rate: {df['Class'].mean()*100:.3f}%")
print(f"Total fraud cases: {df['Class'].sum()}")

os.makedirs('charts', exist_ok=True)

# ── Chart 1: Class Imbalance ─────────────────────────────────
counts = df['Class'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(['Legitimate', 'Fraud'], counts.values, color=['steelblue', 'crimson'])
axes[0].set_title('Class Imbalance: Legitimate vs Fraud')
axes[0].set_ylabel('Number of Transactions')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

axes[1].boxplot(
    [df[df['Class']==0]['Amount'], df[df['Class']==1]['Amount']],
    labels=['Legitimate', 'Fraud']
)
axes[1].set_title('Transaction Amount by Class')
axes[1].set_ylabel('Amount (£)')
axes[1].set_ylim(0, 500)

plt.tight_layout()
plt.savefig('charts/class_imbalance.png', dpi=150)
plt.close()
print("Saved charts/class_imbalance.png")

# ── Chart 2: Fraud Rate by Hour ───────────────────────────────
df['Hour'] = (df['Time'] / 3600 % 24).astype(int)
fraud_by_hour = df[df['Class']==1].groupby('Hour').size()
legit_by_hour = df[df['Class']==0].groupby('Hour').size()
fraud_rate    = (fraud_by_hour / (fraud_by_hour + legit_by_hour) * 100).fillna(0)

plt.figure(figsize=(12, 4))
fraud_rate.plot(kind='bar', color='crimson', alpha=0.8)
plt.title('Fraud Rate by Hour of Day')
plt.xlabel('Hour (0 = midnight)')
plt.ylabel('Fraud Rate (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('charts/fraud_by_hour.png', dpi=150)
plt.close()
print("Saved charts/fraud_by_hour.png")

# ── Feature Engineering ───────────────────────────────────────
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled']   = scaler.fit_transform(df[['Time']])
df_model = df.drop(columns=['Time', 'Amount', 'Hour'])

X = df_model.drop(columns=['Class'])
y = df_model['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ── SMOTE ─────────────────────────────────────────────────────
print("Applying SMOTE...")
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"After SMOTE — Legitimate: {(y_train_sm==0).sum():,} | Fraud: {(y_train_sm==1).sum():,}")

# ── Model ─────────────────────────────────────────────────────
print("\nTraining logistic regression...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_sm, y_train_sm)
y_pred = model.predict(X_test)

recall = recall_score(y_test, y_pred)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
print(f"Fraud Recall: {recall*100:.1f}%")

# ── Chart 3: Confusion Matrix ─────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Legit', 'Predicted Fraud'],
            yticklabels=['Actual Legit', 'Actual Fraud'])
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('charts/confusion_matrix.png', dpi=150)
plt.close()
print("Saved charts/confusion_matrix.png")

# ── Chart 4: Feature Importance ──────────────────────────────
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0],
    'Abs': np.abs(model.coef_[0])
}).sort_values('Abs', ascending=False).head(10)

colors = ['crimson' if c > 0 else 'steelblue' for c in importance_df['Coefficient']]
plt.figure(figsize=(10, 5))
plt.barh(importance_df['Feature'], importance_df['Abs'], color=colors)
plt.xlabel('Absolute Coefficient (Feature Importance)')
plt.title('Top 10 Risk Signals — Logistic Regression')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('charts/feature_importance.png', dpi=150)
plt.close()
print("Saved charts/feature_importance.png")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Transactions analysed : {len(df):,}")
print(f"Fraud cases           : {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
print(f"Imbalance handled via : SMOTE resampling")
print(f"Model                 : Logistic Regression")
print(f"Fraud recall          : {recall*100:.1f}%")
print(f"Charts saved to       : ./charts/")
print("\nTop 3 risk signals:")
for _, row in importance_df.head(3).iterrows():
    direction = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"  {row['Feature']}: {direction} fraud probability")
