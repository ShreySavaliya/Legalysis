# Importing necessary libraries

import pickle
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os


## A function which can predict the type of the unfairness
# def predict_type_of_unfairness(text):

#     model_file = "model_2.pkl"
#     with open(model_file, 'rb') as f:
#         model = pickle.load(f)

#     tfidf_file = "fitted_tfidf_vectorizer_2.pkl"
#     with open(tfidf_file, 'rb') as f:
#         vectorizer = pickle.load(f)

#     sample = vectorizer.transform(text)
#     prediction = model.predict(sample)


    # labels = ["Arbitration", "Unilateral", "Content_removal", "Jurisdiction",
    #           "Choice_of_law", "Limitation_of_liability", "Unilateral_termination",
    #           "Contract_by_using"]
    
#     indices = np.where(prediction == 1)[1]
    
#     predicted_labels = [labels[i] for i in indices]
    
#     return predicted_labels



## A function which can predict whether the term is fair or unfair
def predict_fairness(text):

    file_dir = r'D:\Project\Legal Documents\Programs\app\model\pipelines'
    os.chdir(file_dir)

    pipeline_file = "fairness_check_pipeline.pkl"

    with open(pipeline_file, 'rb') as f:
        pipe = pickle.load(f)

    # sample = tfidf.transform(text)  
    # prediction = model.predict(sample)
        
    prediction = pipe.predict(text)

    if prediction[0] == 1:
        # print("Unfair")
        # response =  predict_type_of_unfairness(text)
        return "Unfair"

    else:
        return "Fair"


def predict_type_of_unfairness(text):

    file_dir = r'D:\Project\Legal Documents\Programs\app\model\pipelines'
    os.chdir(file_dir)

    pipeline_file = "unfairness_datails_pipeline.pkl"

    with open(pipeline_file, 'rb') as f:
        pipe = pickle.load(f)

    prediction = pipe.predict(text)

    labels = ["Arbitration", "Unilateral", "Content_removal", "Jurisdiction",
              "Choice_of_law", "Limitation_of_liability", "Unilateral_termination",
              "Contract_by_using"]

    return prediction