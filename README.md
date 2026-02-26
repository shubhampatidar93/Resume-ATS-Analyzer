# AI-Resume-Builder

A simple AI-powered resume builder and ATS scorer.

## Project Structure
AI-Resume-Builder/
├── backend/
│   ├── app.py
│   ├── ml_model.py
│   ├── resume_generator.py
│   ├── requirements.txt
│   └── templates/
│       └── resume_template.html
├── frontend/
│   ├── index.html
│   ├── result.html
│   └── style.css
├── dataset/
│   └── resume_dataset.csv
└── README.md

## How To Run

### Step 1: Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### Step 2: Train Model
Open a Python terminal and run:
```python
from backend.ml_model import ATSModel
model = ATSModel()
model.train("dataset/resume_dataset.csv")
```

### Step 3: Run Backend
```bash
python backend/app.py
```

### Step 4: Open Frontend
Open `frontend/index.html` in your browser.
