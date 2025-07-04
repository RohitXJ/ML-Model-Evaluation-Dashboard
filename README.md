# 🧠 AI Engineer Level 2 — Project D
## 📊 ML Model Evaluation Dashboard (Scripted)

An interactive, modular Python script that:
- Loads a CSV dataset via user input
- Lets users select ML models to compare
- Automatically builds pipelines per model
- Cross-validates and compares metrics (accuracy, precision, recall, F1, ROC AUC)
- Shows + optionally saves plots and summary

---

## ✅ TODO LIST — BUILDING CHECKPOINTS
Use this list to track your implementation step-by-step. Each block can be completed/tested independently.

---

### 🔹 STEP 0: Project Folder Structure
```
model_evaluator/
├── main.py                # main script with all user I/O logic
├── models.py              # contains available model definitions
├── preprocessing.py       # function to build data preprocessor
├── evaluator.py           # handles cross-validation and metric computation
├── plotter.py             # all visualizations (bar, ROC, confusion)
└── utils.py               # helpers (e.g. user input menu)
```
☐ Create folder and empty files as above.

---

### 🔹 STEP 1: User Input Interface (in `main.py`)
- [X] Prompt for CSV file path
- [X] Prompt for target column name
- [X] Load the dataset with pandas
- [X] Validate if target column exists
- [X] Prompt for model selection (from numbered list)
- [X] Store selected model keys

---

### 🔹 STEP 2: Preprocessing (in `preprocessing.py`)
- [X] Detect numerical and categorical columns
- [X] Build `ColumnTransformer` with:
  - SimpleImputer(median) + StandardScaler for numeric
  - SimpleImputer(mode) + OneHotEncoder for categorical
- [X] Return the complete preprocessor pipeline

---

### 🔹 STEP 3: Available Models (in `models.py`)
- [ ] Define `get_available_models()` returning a dict like:
```python
{
  "1": ("Logistic Regression", LogisticRegression(...)),
  "2": ("Random Forest", RandomForestClassifier(...)),
  "3": ("SVM", SVC(...)),
  ...
}
```
- [ ] Consider reasonable defaults (max_iter, n_estimators=100, etc.)
- [ ] Use `n_jobs=-1` wherever possible

---

### 🔹 STEP 4: Build Dynamic Pipelines (in `main.py`)
- [ ] Using selected model keys, build pipelines with:
```python
Pipeline([
  ("preprocess", preprocessor),
  ("model", selected_model_object)
])
```
- [ ] Store as: `model_name -> pipeline`

---

### 🔹 STEP 5: Evaluation Logic (in `evaluator.py`)
- [ ] Accept model pipelines and training data
- [ ] For each model, perform StratifiedKFold cross-validation
- [ ] Compute mean of: accuracy, precision, recall, F1, ROC AUC
- [ ] Return summary as a pandas DataFrame

---

### 🔹 STEP 6: Plotting (in `plotter.py`)
- [ ] Bar plot for any selected metric (e.g. accuracy)
- [ ] Combined ROC Curve
- [ ] Confusion Matrix (for top model or each model)
- [ ] Use matplotlib / seaborn

---

### 🔹 STEP 7: Final Integration (in `main.py`)
- [ ] Perform train/test split
- [ ] Build pipelines
- [ ] Evaluate models
- [ ] Display table
- [ ] Ask user: save CSV + plots?
- [ ] Save if yes into `./output` folder

---

### 🔹 BONUS (Optional Extensions)
- [ ] Add GridSearchCV for optional tuning
- [ ] Auto-downsample if dataset > 10k rows
- [ ] Allow exclusion of low-importance features
- [ ] Export entire session result as HTML report

---

## ✅ HOW TO RUN (later)
```bash
python main.py
```

---

Stay consistent. Each script file should do one thing well. Ask me anytime if you get stuck on a step — we’ll code it together 💻🚀

