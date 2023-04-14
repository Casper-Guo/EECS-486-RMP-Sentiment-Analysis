import re
from joblib import load
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier as XGBoost
from sklearn.model_selection import train_test_split
import tkinter as tk  
from tkinter import ttk

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('omw-1.4')

def preprocess_comment(row):
    """
    Tokenize, remove stopwords and punctuations at word ends, lemmatize, and then reassemble into one string.
    
    If any token matches the first or last name of the professor, it is dropped.
    
    All numbers are dropped.
    
    This is done to eliminate low-impact tokens and reduce vocabulary size.
    
    String type output required for easier ingestion by sklearn TfidfVectorizer.
    """
    comment = row.loc["comment"]
    re.sub(r"['!\"#$%&\'()*,./:;<=>?@[\\]^_`{|}~'] ", ' ', comment)
    tokens = word_tokenize(comment)
    
    ignore_list = stopword_list.copy()
    ignore_list.update([row.loc["firstName"], row.loc["lastName"]])
    
    tokens = [token.lower() for token in tokens if token not in ignore_list and not token.isnumeric()]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return " ".join(tokens)

###############
# Ingest Data #
###############

df_ratings = pd.read_csv("Data/clean_ratings.csv", infer_datetime_format=True)

df_profs = pd.read_csv("Data/clean_prof_info.csv")
df_profs["firstName"] = df_profs["firstName"].apply(lambda x: x.strip())
df_profs["lastName"] = df_profs["lastName"].apply(lambda x: x.strip())

df_names = df_profs[["profID", "firstName", "lastName"]]
df_ratings = df_ratings.merge(df_names, how="inner", on="profID")

stopword_list = set(stopwords.words("english"))
stopword_list.update([',', '.'])
lemmatizer = WordNetLemmatizer()

df_ratings["comment"] = df_ratings.apply(preprocess_comment, axis=1)

df_ratings["Hot"] = df_ratings["helpfulRating"] >= 4
df_ratings["Hot"] = df_ratings["Hot"].astype(int)

X, y = df_ratings["comment"], df_ratings["Hot"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

####################
# Train Vectorizer #
####################
tfidf_vectorizer = TfidfVectorizer(strip_accents="ascii")
tfidf_vectorizer.fit(X_train)

#######################
# Load Trained Models #
#######################

logistic_regression = load("Models/log_reg_tfidf.joblib")
random_forest = load("Models/rf_tfidf.joblib")
svm = load("Models/svm_tfidf.joblib")

##################
# Model Training #
##################
'''
Version conflict requires retraining XGBoost
'''
train_tfidf = tfidf_vectorizer.transform(X_train)
test_tfidf = tfidf_vectorizer.transform(X_test)

# Hyperparameters selected by prior grid search
xgboost = XGBoost(learning_rate=0.5, n_estimators=200, random_state=42)
xgboost.fit(train_tfidf, y_train)

#######
# GUI #
#######

class demo_gui:
    def __init__(self, root):
        
        root.title("Interactive Demo")

        mainframe = ttk.Frame(root, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.text = tk.StringVar()
        text_entry = ttk.Entry(mainframe, width=100, textvariable=self.text)
        text_entry.grid(column=2, row=1, sticky=(tk.W, tk.E))

        ttk.Button(mainframe, text="Predict", command=self.predict).grid(column=3, row=1, sticky=(tk.W, tk.E))

        ttk.Label(mainframe, text="Enter a comment").grid(column=1, row=1, sticky=(tk.W, tk.E))
        ttk.Label(mainframe, text="Processed Comment").grid(column=1, row=2, sticky=(tk.W, tk.E))
        ttk.Label(mainframe, text="Random Forest").grid(column=1, row=3, sticky=(tk.W, tk.E))
        ttk.Label(mainframe, text="XGBoost").grid(column=1, row=4, sticky=(tk.W, tk.E))
        ttk.Label(mainframe, text="Logistic Regression").grid(column=1, row=5, sticky=(tk.W, tk.E))
        ttk.Label(mainframe, text="Support Vector Machine").grid(column=1, row=6, sticky=(tk.W, tk.E))
        
        self.tokens = tk.StringVar()
        ttk.Label(mainframe, textvariable=self.tokens).grid(column=2, row=2, sticky=(tk.W, tk.E))
        
        self.rf_pred = tk.StringVar()
        ttk.Label(mainframe, textvariable=self.rf_pred).grid(column=2, row=3, sticky=(tk.W, tk.E))
        
        self.xgb_pred = tk.StringVar()
        ttk.Label(mainframe, textvariable=self.xgb_pred).grid(column=2, row=4, sticky=(tk.W, tk.E))

        self.lgr_pred = tk.StringVar()
        ttk.Label(mainframe, textvariable=self.lgr_pred).grid(column=2, row=5, sticky=(tk.W, tk.E))

        self.svm_pred = tk.StringVar()
        ttk.Label(mainframe, textvariable=self.svm_pred).grid(column=2, row=6, sticky=(tk.W, tk.E))
        
        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=10, pady=5)

        text_entry.focus()
    
    def predict(self):
        tokens, matrix = self.process_input()
        
        if not tokens:
            self.tokens.set('')
            self.rf_pred.set('')
            self.xgb_pred.set('')
            self.lgr_pred.set('')
            self.svm_pred.set('')
            
            return
        
        self.tokens.set(tokens)
        self.rf_pred.set(self.random_forest_predict(random_forest, matrix))
        self.xgb_pred.set(self.xgboost_predict_proba(xgboost, matrix))
        self.lgr_pred.set(self.logistic_regression_predict(logistic_regression, matrix))
        self.svm_pred.set(self.support_vector_machine_predict(svm, matrix))
    

    def process_input(self):
        text = self.text.get()
        re.sub(r"['!\"#$%&\'()*,./:;<=>?@[\\]^_`{|}~'] ", ' ', text)
        tokens = word_tokenize(text)

        tokens = [token.lower() for token in tokens if token not in stopword_list and not token.isnumeric()]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        tokens = ' '.join(tokens)

        return tokens, tfidf_vectorizer.transform([tokens])
    
    
    def random_forest_predict(self, clf, encoding):
        pred = clf.predict_proba(encoding)
        
        return f"Hot probability: {round(pred[0, 1], 2)}, not hot probability: {round(pred[0, 0], 2)}"
    
    
    def logistic_regression_predict(self, clf, encoding):
        pred = clf.predict_proba(encoding)
        
        return f"Hot probability: {round(pred[0, 1], 2)}, not hot probability: {round(pred[0, 0], 2)}"
    
    
    def support_vector_machine_predict(self, clf, encoding):
        pred = clf.predict(encoding)
        hot_or_not = "Hot" if pred[0] else "Not hot"
        
        return f"Predicted class: {hot_or_not}"
    
    
    def xgboost_predict_proba(self, clf, encoding):
        pred = clf.predict_proba(encoding)
        
        return f"Hot probability: {round(pred[0, 1], 2)}, not hot probability: {round(pred[0, 0], 2)}"
    

def main():
    root = tk.Tk()
    demo_gui(root)
    root.mainloop()


if __name__ == "__main__":
    main()