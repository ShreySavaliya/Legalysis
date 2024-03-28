import os
import pandas as pd
import numpy as np
import pickle

file_dir = r'D:\Project\Legal Documents\Programs\app\model\pipelines'
os.chdir(file_dir)

with open('fairness_check_pipeline.pkl', 'rb') as f:
    pipe = pickle.load(f)

text = ["you consent to receive communications from us electronically , such as e-mails , texts , mobile push notices , or notices and messages on this site or through the other amazon services , such as our message center , and you can retain copies of these communications for your records . "]
output = pipe.predict(text)
print(output)
print(type(output))
output.tolist()
print(output)