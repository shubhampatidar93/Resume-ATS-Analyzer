"""
Machine Learning model for Resume ATS Analysis.
Uses TF-IDF Vectorization and Linear Regression to predict ATS scores
based on the similarity between a resume and a job description.
"""
import json
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression


class ATSModel:
    """
    A class to represent the ATS Model for resume analysis.
    """

    def __init__(self):
        """Initializes the TF-IDF vectorizer and Linear Regression model."""
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=8000)
        self.model = LinearRegression()

    def clean_text(self, text):
        """
        Cleans input text by lowercasing and removing non-alphabetic characters.
        
        Args:
            text (str): The raw text to clean.
            
        Returns:
            str: The cleaned text.
        """
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z ]', '', text)
        return text

    def load_jsonl(self, file_path):
        """
        Loads and cleans data from a JSONL dataset file.
        
        Args:
            file_path (str): Path to the .jsonl file.
            
        Returns:
            tuple: A list of combined resume+job texts and a numpy array of scores.
        """
        resumes = []
        scores = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                resume = self.clean_text(data["resume_text"])
                job = self.clean_text(data["job_description"])
                score = data["ats_score"]

                # Combine resume and job text for vectorization
                resumes.append(resume + " " + job)
                scores.append(score)

        return resumes, np.array(scores)

    def train(self, file_path):
        """
        Trains the model using the provided dataset and saves it to disk.
        
        Args:
            file_path (str): Path to the training dataset.
        """

        texts, scores = self.load_jsonl(file_path)

        # Vectorize the text data
        X = self.vectorizer.fit_transform(texts)

        # Train the linear regression model
        self.model.fit(X, scores)

        # Ensure the backend directory exists for saving models
        import os
        if not os.path.exists("backend"):
            os.makedirs("backend")

        # Save the trained model and vectorizer
        pickle.dump(self.model, open("backend/ats_model.pkl", "wb"))
        pickle.dump(self.vectorizer, open("backend/vectorizer.pkl", "wb"))

        print("Model trained successfully on JSONL dataset!")

    def load(self):
        """Loads the trained model and vectorizer from the disk."""
        self.model = pickle.load(open("backend/ats_model.pkl", "rb"))
        self.vectorizer = pickle.load(open("backend/vectorizer.pkl", "rb"))

    def predict(self, resume, job_desc):
        """
        Predicts the ATS score for a given resume against a job description.
        
        Args:
            resume (str): The resume text.
            job_desc (str): The job description text.
            
        Returns:
            tuple: (Score, List of Matched Keywords, List of Missing Keywords)
        """

        resume_clean = self.clean_text(resume)
        job_clean = self.clean_text(job_desc)

        combined = resume_clean + " " + job_clean

        # Transform the input text using the loaded vectorizer
        vector = self.vectorizer.transform([combined])

        # Predict score and bound it between 0 and 100
        score = self.model.predict(vector)[0]
        score = max(0, min(100, round(score, 2)))

        # Perform keyword analysis
        matching, missing = self.keyword_analysis(resume_clean, job_clean)

        return score, matching, missing

    def keyword_analysis(self, resume, job_desc):
        """
        Identifies matching and missing keywords between the resume and job description.
        
        Args:
            resume (str): Cleaned resume text.
            job_desc (str): Cleaned job description text.
            
        Returns:
            tuple: (List of matched words, List of missing words)
        """
        resume_words = set(resume.split())
        jd_words = set(job_desc.split())

        # Find intersection and difference of word sets
        matching = list(resume_words.intersection(jd_words))
        missing = list(jd_words.difference(resume_words))

        # Limit to top 20 results for display
        return matching[:20], missing[:20]
