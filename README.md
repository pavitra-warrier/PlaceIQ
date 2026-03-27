# PlaceIQ — Smart Placement Guidance System

## ⚡ Quick Start (3 steps)

### Step 1 — Install dependencies
```bash
pip install flask scikit-learn numpy joblib
```

### Step 2 — Generate the ML model (run ONCE)
```bash
python train_model.py
```
You should see: `✅ Models saved: placement_model.pkl, placement_scaler.pkl`

### Step 3 — Start the Flask server
```bash
python app.py
```
You should see: `🚀 Starting Smart Placement Guidance System...`

### Step 4 — Open your browser
Go to: **http://localhost:5000**

---

## 📁 File Structure
```
placement_project/
├── train_model.py          ← Run this once to create model files
├── app.py                  ← Flask backend server
├── index.html              ← Frontend UI (served by Flask)
├── placement_model.pkl     ← Generated after train_model.py
└── placement_scaler.pkl    ← Generated after train_model.py
```

## ❓ Common Errors

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: flask` | Run `pip install flask scikit-learn numpy joblib` |
| `FileNotFoundError: placement_model.pkl` | Run `python train_model.py` first |
| `Connection refused` on frontend | Make sure `python app.py` is running |
| Port 5000 already in use | Change `port=5000` to `port=5001` in app.py |

## 🎯 Features
- Predicts placement likelihood (Yes/No) with confidence
- Identifies best-fit role: SDE, Data Analyst, ML Engineer, Core Engineering, Non-Tech
- Shows role fit distribution across all categories
- Provides personalized tips per role
- Displays average salary package for predicted role
