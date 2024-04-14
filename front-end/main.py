from flask import Flask, render_template, request
import requests
import json
app = Flask(__name__)

def clean_data(original_json):
    # Step 1: Replace single quotes
    modified_json = original_json.replace("'", "\\'")

    # # Step 2: Remove unnecessary strings
    unwanted_strings = ['-lrb-', '-rrb-', '-lcb-', '-rcb-']
    for string in unwanted_strings:
        modified_json = modified_json.replace(string, '')

    # # Step 3: Remove line breaks
    modified_string = modified_json.replace('\n', '').strip()


    output_dict = {"text": modified_json}

    print(output_dict)

    return output_dict




def check(inputData):
    url = 'http://127.0.0.1:8000/predict'
    response = requests.post(url,json=inputData)

@app.route('/')
def startPage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    inputData = str(request.form['user_input'])

    string_data = clean_data(inputData)

    print("****************************")
    # print(type(string_data['text']))
    print(string_data)
    print("****************************")

    url = 'http://127.0.0.1:8000/predict'

    response = requests.post(url,json=string_data)

    response_dict = response.json()

    print(response_dict)

    if "fairness" in response_dict:
        return "The terms and conditions are fair"
    

    labels = [
        "Arbitration", "Unilateral", "Content_removal", "Jurisdiction",
        "Choice_of_law", "Limitation_of_liability", "Unilateral_termination",
        "Contract_by_using"
    ]

    # Iterate over the dictionary items
    for key, value in response_dict.items():
        # Get the predicted_labels list
        predicted_labels = value["predicted_labels"][0]

        # Iterate over the elements of predicted_labels list
        for i, label in enumerate(predicted_labels):
            # If the label is 1, update it with the corresponding label from the 'labels' list
            if label == 1:
                predicted_labels[i] = labels[i]
            # If the label is 0, remove it from the list
            else:
                predicted_labels[i] = None

        # Filter out the None values
        value["predicted_labels"][0] = [label for label in predicted_labels if label is not None]

        # If the predicted_labels list is empty, add "Others"
        if not value["predicted_labels"][0]:
            value["predicted_labels"][0].append("Others")

        
    
    return render_template("final.html", response_dict = response_dict)


if __name__ == '__main__':
    app.run(debug=True)