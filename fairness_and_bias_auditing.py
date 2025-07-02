import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

# ---------------------------------
# Generate synthetic data
# ---------------------------------
np.random.seed(0)
n_samples = 1000
X = np.random.randn(n_samples, 2)
protected_attr = np.random.choice([0, 1], size=n_samples)  # 0 = group A, 1 = group B

# Introduce bias
prob = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1] - 0.7 * protected_attr)))
y = np.random.binomial(1, prob)

# Combine into DataFrame
df = pd.DataFrame(X, columns=['feature1', 'feature2'])
df['protected'] = protected_attr
df['target'] = y

# Train/test split
X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
    df[['feature1', 'feature2']], df['target'], df['protected'], test_size=0.3, random_state=42
)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Overall metrics
overall_acc = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {overall_acc:.2f}")

# Group-wise metrics
groups = np.unique(prot_test)
results = []
for group in groups:
    idx = (prot_test == group)
    acc = accuracy_score(y_test[idx], y_pred[idx])
    tpr = recall_score(y_test[idx], y_pred[idx])
    pos_rate = np.mean(y_pred[idx])
    results.append({'group': group, 'accuracy': acc, 'TPR': tpr, 'positive_rate': pos_rate})

df_results = pd.DataFrame(results)
print("\nGroup-wise metrics:\n", df_results)

# Compute disparate impact
pr_0 = df_results[df_results['group'] == 0]['positive_rate'].values[0]
pr_1 = df_results[df_results['group'] == 1]['positive_rate'].values[0]
disparate_impact = pr_1 / pr_0
print(f"\nDisparate Impact (group 1 / group 0): {disparate_impact:.2f}")

# Flag potential bias
if disparate_impact < 0.8 or disparate_impact > 1.25:
    print("⚠️ Potential bias detected: Disparate impact outside acceptable range (0.8 - 1.25).")
else:
    print("✅ No major disparate impact detected.")
