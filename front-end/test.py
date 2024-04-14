# import json

# def clean_data(original_json):
#     # Step 1: Replace single quotes
#     modified_json = original_json.replace("'", "\\'")

#     # Step 2: Remove unnecessary strings
#     unwanted_strings = ['-lrb-', '-rrb-', '-lcb-', '-rcb-']
#     for string in unwanted_strings:
#         modified_json = modified_json.replace(string, '')

#     # Step 3: Remove line breaks
#     modified_json = modified_json.replace('\n', '')

#     # Print modified JSON
#     print("********************")
    #     print(type(modified_json))
    #     print("********************")
    
    #     return modified_json
    
    
    
    # a = input("ENter the string:")
    
    # clean_data(a)
    
    
# response_dict = {
#       "1": {
#         "sentence": "by accessing or using any of the services , you agree to be bound by these terms .",
#         "fairness": 1,
#         "predicted_labels": [
#           [
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             1
#           ]
#         ]
#       },
#       "2": {
#         "sentence": "notwithstanding any other remedies available to truecaller , you agree that truecaller may suspend or terminate your use of the services without notice if you use the services or the content in any prohibited manner , and that such use will be deemed a material breach of these terms .",
#         "fairness": 1,
#         "predicted_labels": [
#           [
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0
#           ]
#         ]
#       },
#       "3": {
#         "sentence": "any such unmoderated content will not be covered by constitutional protection under the swedish fundamental law on freedom of expression as per the paragraph above and truecaller will not be liable for the publication of such content .",
#         "fairness": 1,
#         "predicted_labels": [
#           [
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0
#           ]
#         ]
#       },
#       "4": {
#         "sentence": "you are solely liable for any content that you post or communicate directly in or on the services and you agree to only post or communicate content that : a.",
#         "fairness": 1,
#         "predicted_labels": [
#           [
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0
#           ]
#         ]
#       },
#       "5": {
#         "sentence": "truecaller hereby reserves the right in its absolute discretion to remove any user generated content from the services .",
#         "fairness": 1,
#         "predicted_labels": [
#           [
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0
#           ]
#         ]
#       },
#       "6": {
#         "sentence": "such third party content , websites and services may be subject to the respective third party terms and conditions and truecaller will not be liable for any such third party content , websites or services .",
#         "fairness": 1,
#         "predicted_labels": [
#           [
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0,
#             0
#           ]
#         ]
#       },
#       "7": {
#     "sentence": "you understand and acknowledge that the services may be unavailable from time to time and that truecaller will not be liable for your inability to use the services for whatever reason .",
#     "fairness": 1,
#     "predicted_labels": [
#       [
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0
#       ]
#     ]
#   },
#   "8": {
#     "sentence": "to the maximum extent permitted by applicable law , you expressly agree that truecaller shall in no event be liable for any direct , indirect , special , incidental , consequential or exemplary damages , including but not limited to damages for loss of profits , data and goodwill , arising out of the use or inability to use the services or the content , even if advised of the possibility of such damages .",
#     "fairness": 1,
#     "predicted_labels": [
#       [
#         0,
#         0,
#         0,
#         0,
#         0,
#         1,
#         0,
#         0
#       ]
#     ]
#   },
#   "9": {
#     "sentence": "in particular , and without limitation , truecaller shall have no liability for any information stored or processed within the services , including the costs of recovering such information .",
#     "fairness": 1,
#     "predicted_labels": [
#       [
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0
#       ]
#     ]
#   },
#   "10": {
#     "sentence": "truecaller shall not be liable for the validity , reliability or correctness of the content and information provided through and in connection with use of the services .",
#     "fairness": 1,
#     "predicted_labels": [
#       [
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0
#       ]
#     ]
#   },
#   "11": {
#     "sentence": "we may change the truecaller fees from time to time by posting the changes in or on the services or by notifying you in advance .",
#     "fairness": 1,
#     "predicted_labels": [
#       [
#         0,
#         1,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0
#       ]
#     ]
#   },
#   "12": {
#     "sentence": "failure to pay these fees may result in suspension or termination of your service or subscription .",
#     "fairness": 1,
#     "predicted_labels": [
#       [
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0
#       ]
#     ]
#   },
#   "13": {
#     "sentence": "truecaller may terminate the terms and your use of the services at any time with thirty -lrb- 30 -rrb- days ' advance notice .",
#     "fairness": 1,
#     "predicted_labels": [
#       [
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0
#       ]
#     ]
#   },
#   "14": {
#     "sentence": "truecaller also reserves the right to modify these terms at any time by providing revised terms to the user or by publishing the revised terms within the services .",
#     "fairness": 1,
#     "predicted_labels": [
#       [
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0
#       ]
#     ]
#   },
#   "15": {
#     "sentence": "any continued use by you of the services following publication or notification of revised terms shall constitute your acceptance to the revised terms .",
#     "fairness": 1,
#     "predicted_labels": [
#       [
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0
#       ]
#     ]
#   },
#   "16": {
#     "sentence": "these terms shall be governed and construed in accordance with the laws of sweden , excluding its choice of law rules .",
#     "fairness": 1,
#     "predicted_labels": [
#       [
#         0,
#         0,
#         0,
#         0,
#         1,
#         0,
#         0,
#         0
#       ]
#     ]
#   },
#   "17": {
#     "sentence": "all disputes arising out of or in connection with these terms shall be adjudicated exclusively in sweden , with the district court of stockholm as the court of first instance .",
#     "fairness": 1,
#     "predicted_labels": [
#       [
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0
#       ]
#     ]
#   }
# }

# labels = ["Arbitration", "Unilateral", "Content_removal", "Jurisdiction",
#             "Choice_of_law", "Limitation_of_liability", "Unilateral_termination",
#             "Contract_by_using"]

# length_of_labels = len(labels)

# print(length_of_labels)

# for item in response_dict:
#     print(type(item["predicted_labels"][0][1]))
#     print(item["predicted_labels"][0][1])



# for item in response_dict:
#     for j in range(len(labels)):

#         if int((response_dict[item]["predicted_labels"][0][j])) == 1:
#             (response_dict[item]["predicted_labels"][0][j]) = labels[j]





# for key, value in response_dict.items():
#     # Check if any '1' exists in the predicted_labels
#     if 1 in value["predicted_labels"][0]:
#         # Replace the '1' with corresponding label from the labels list
#         modified_labels = [labels[i] if v == 1 else 0 for i, v in enumerate(value["predicted_labels"][0])]
#         response_dict[key]["predicted_labels"] = modified_labels

# # Print the modified response_dict
# print(response_dict)



response_dict = {
    "1": {
        "sentence": "by accessing or using any of the services , you agree to be bound by these terms .",
        "fairness": 1,
        "predicted_labels": [
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1
            ]
        ]
    },
    "2": {
        "sentence": "notwithstanding any other remedies available to truecaller , you agree that truecaller may suspend or terminate your use of the services without notice if you use the services or the content in any prohibited manner , and that such use will be deemed a material breach of these terms .",
        "fairness": 1,
        "predicted_labels": [
            [
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0
            ]
        ]
    },
    "3": {
        "sentence": "any such unmoderated content will not be covered by constitutional protection under the swedish fundamental law on freedom of expression as per the paragraph above and truecaller will not be liable for the publication of such content .",
        "fairness": 1,
        "predicted_labels": [
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ]
        ]
    },
    "4": {
        "sentence": "you are solely liable for any content that you post or communicate directly in or on the services and you agree to only post or communicate content that : a.",
        "fairness": 1,
        "predicted_labels": [
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ]
        ]
    },
    "5": {
        "sentence": "truecaller hereby reserves the right in its absolute discretion to remove any user generated content from the services .",
        "fairness": 1,
        "predicted_labels": [
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ]
        ]
    }
}

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

print(response_dict)
