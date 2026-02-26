"""
Flask backend for the Resume ATS Analyzer.
Handles file uploads, text extraction from PDFs, and communicates with the ML model.
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ml_model import ATSModel
import PyPDF2
import os

# Initialize Flask app with template and static folders pointing to the frontend directory
app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
CORS(app)

# Initialize and load the ATS model
model = ATSModel()
try:
    model.load()
except Exception as e:
    print(f"Model not found: {e}. Please train the model first.")

def extract_pdf_text(file):
    """
    Extracts text from a PDF file or reads a text file.
    
    Args:
        file: The uploaded file object.
        
    Returns:
        str: The extracted text content.
    """
    if file.filename.endswith('.pdf'):
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    else:
        # Assume it's a text file and decode it
        return file.read().decode('utf-8')


@app.route("/")
def index():
    """Renders the main interface."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Handles the ATS analysis request.
    Extracts resume content from file or text area, compares it with the job description,
    and returns the score and keyword analysis.
    """
    job_desc = request.form.get("job_description")
    input_method = request.form.get("input_method")
    
    resume_text = ""
    
    # Handle resume input based on the selected method
    if input_method == 'file':
        if "resume_file" in request.files:
            file = request.files["resume_file"]
            if file.filename != '':
                resume_text = extract_pdf_text(file)
            else:
                return "No file selected", 400
        else:
            return "No resume file part", 400
    else:
        resume_text = request.form.get("resume_text")

    # Basic validation
    if not resume_text or not job_desc:
        return "Missing resume text or job description", 400

    # Get predictions from the ML model
    score, matching, missing = model.predict(resume_text, job_desc)

    # Return results to the main page
    return render_template("index.html", 
                           ats_score=score, 
                           matched=matching, 
                           missing=missing)


if __name__ == "__main__":
    app.run(debug=True)
