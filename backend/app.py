# from fastapi import FastAPI
# import uvicorn
# from pydantic import BaseModel
# import pickle
# import os
# import numpy as np

# app = FastAPI()

# class TextIn(BaseModel):
#     text: list

# file_dir = r'D:\Project\legalysis\legalysis\Programs\app\model\pipelines'
# os.chdir(file_dir)

# pipeline_file = "fairness_check_pipeline.pkl"

# with open(pipeline_file, 'rb') as f:
#     pipe = pickle.load(f)

# labels = ["Arbitration", "Unilateral", "Content_removal", "Jurisdiction",
#             "Choice_of_law", "Limitation_of_liability", "Unilateral_termination",
#             "Contract_by_using"]

# def predict_type(text):
#     os.chdir(file_dir)

#     with open('unfairness_details_pipeline.pkl', 'rb') as f:
#         pipe = pickle.load(f)

#     pred = pipe.predict(text)
#     pred_list = pred.tolist()
#     return {"fairness_type": pred_list}


# @app.get('/')
# def home():
#     return {"Hello": "world"}

# @app.post('/predict')
# def predict(payload: TextIn):
#     prediction = int(pipe.predict(payload.text))

#     if prediction == 1:
#         output = predict_type(payload.text)
#         a = output["fairness_type"]

#         return {
#                 "fairness": prediction,
#                 "predicted_labels": a}

#     return {"fairness": prediction}

# # url : https://127.0.0.1:8080/predict
# if __name__=="__main__":
#     uvicorn.run(app, host='127.0.0.1', port=8000)











from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pickle
import os
import numpy as np

app = FastAPI()
def separate_paragraph(paragraph):
    sentences = []
    
    start_index = 0
    
    for i, char in enumerate(paragraph):
        if char == '.':
            sentence = paragraph[start_index:i+1].strip()
            sentence_list = []
            sentence_list.append(sentence)
            sentences.append(sentence_list)
            start_index = i + 1
    
    if start_index < len(paragraph):
        last_sentence = paragraph[start_index:].strip()
        sentences.append([last_sentence])  # Ensure it's a list
        
    return sentences    



class TextIn(BaseModel):
    text: str

file_dir = r'D:\Project\MiniProjectLegalysis\backend\model\pipelines'
os.chdir(file_dir)

pipeline_file = "fairness_check_pipeline.pkl"

with open(pipeline_file, 'rb') as f:
    pipe = pickle.load(f)

labels = ["Arbitration", "Unilateral", "Content_removal", "Jurisdiction",
            "Choice_of_law", "Limitation_of_liability", "Unilateral_termination",
            "Contract_by_using"]

def predict_type(text):
    os.chdir(file_dir)

    with open('unfairness_details_pipeline.pkl', 'rb') as f:
        pipe = pickle.load(f)

    pred = pipe.predict(text)
    pred_list = pred.tolist()
    return {"fairness_type": pred_list}


@app.get('/')
def home():
    return {"Hello": "world"}

@app.post('/predict')
def predict(payload: TextIn):

    list_of_sentences = separate_paragraph(payload.text)

    list_for_return = {}
    id = 1
    for i in list_of_sentences:
        
        print(type(i))

        prediction = int(pipe.predict(i))

        if prediction == 1:
            output = predict_type(i)
            a = output["fairness_type"]
            list_for_return[id] = {
                    "sentence":i[0],
                    "fairness": prediction,
                    "predicted_labels": a
                    }

            id+=1

    if len(list_for_return) == 0:
        return {"fairness": "Completely fair"}
    

    return list_for_return

# url : https://127.0.0.1:8080/predict
if __name__=="__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)