# Resume ATS Analyzer

## Overview
Resume ATS Analyzer is a project designed to analyze resumes and evaluate their compatibility with Applicant Tracking Systems (ATS). The project leverages machine learning to parse and score resumes based on predefined criteria.

## Project Structure
```
backend/
    app.py                # Backend application logic
    ml_model.py           # Machine learning model for resume analysis
    requirements.txt      # Python dependencies
    templates/
        resume_template.html  # HTML template for resume rendering

dataset/
    resume_dataset.jsonl  # Dataset for training/testing the model

frontend/
    static/
        style.css         # CSS for frontend styling
    templates/
        index.html        # Main HTML file for the frontend
```

## Features
- Parse resumes and extract key information.
- Evaluate resumes against ATS criteria.
- Provide feedback to improve resume compatibility.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd Resume ATS Analyzer
   ```
3. Install the required Python packages:
   ```bash
   pip install -r backend/requirements.txt
   ```

## Usage
1. Start the backend server:
   ```bash
   python backend/app.py
   ```
2. Open `index.html` in the `frontend/templates/` directory to access the frontend interface.

## Dataset
The dataset used for training and testing the model is located in the `dataset/` directory as `resume_dataset.jsonl`.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Thanks to all contributors and open-source libraries used in this project.