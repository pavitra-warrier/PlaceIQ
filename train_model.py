"""
train_model.py — Run this ONCE to generate placement_model.pkl and placement_scaler.pkl
Usage: python train_model.py
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

np.random.seed(42)
N = 1000

# Simulate realistic student data
cgpa               = np.random.uniform(5.0, 10.0, N)
backlogs           = np.random.randint(0, 5, N)
semester           = np.random.randint(5, 9, N)
leetcode_rating    = np.random.uniform(0, 3000, N)
codechef_rating    = np.random.uniform(0, 2500, N)
contests_won       = np.random.randint(0, 10, N)
hackathons         = np.random.randint(0, 8, N)
internships        = np.random.randint(0, 4, N)
projects_count     = np.random.randint(0, 10, N)
communication_score= np.random.uniform(1, 10, N)
leadership_score   = np.random.uniform(1, 10, N)
ds_score           = np.random.uniform(1, 10, N)
ml_score           = np.random.uniform(1, 10, N)
web_score          = np.random.uniform(1, 10, N)
embedded_score     = np.random.uniform(1, 10, N)

X = np.column_stack([
    cgpa, backlogs, semester, leetcode_rating, codechef_rating,
    contests_won, hackathons, internships, projects_count,
    communication_score, leadership_score, ds_score, ml_score,
    web_score, embedded_score
])

# Placement label: 1 if overall profile is strong
strength = (
    cgpa / 10 * 0.25 +
    (1 - backlogs / 5) * 0.10 +
    leetcode_rating / 3000 * 0.20 +
    internships / 4 * 0.15 +
    projects_count / 10 * 0.10 +
    communication_score / 10 * 0.10 +
    (contests_won + hackathons) / 18 * 0.10
)
placement_label = (strength + np.random.normal(0, 0.05, N) > 0.45).astype(int)

# Role label based on skill scores
# 0=SDE, 1=Data Analyst, 2=ML Engineer, 3=Core Engineering, 4=Non-Tech
role_scores = np.column_stack([
    (ds_score + web_score + leetcode_rating / 300),          # SDE
    (ds_score * 1.5 + ml_score),                             # Data Analyst
    (ml_score * 1.5 + ds_score),                             # ML Engineer
    (embedded_score + cgpa),                                 # Core Engineering
    (communication_score + leadership_score),                # Non-Tech
])
role_label = np.argmax(role_scores + np.random.normal(0, 0.5, role_scores.shape), axis=1)

y = np.column_stack([placement_label, role_label])

# Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf.fit(X_scaled, y)

joblib.dump(clf, "placement_model.pkl")
joblib.dump(scaler, "placement_scaler.pkl")
print("✅ Models saved: placement_model.pkl, placement_scaler.pkl")
