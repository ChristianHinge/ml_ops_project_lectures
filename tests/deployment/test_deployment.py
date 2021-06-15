import requests
import json
import torch
import numpy as np

x_new = np.zeros((1,1,28,28))
# Convert the array to a serializable list in a JSON document
input_json = json.dumps({"data": x_new.tolist()})

# Set the content type
headers = { 'Content-Type':'application/json' }

predictions = requests.post("http://127.0.0.1:1111/score", input_json, headers = headers)
print(predictions)
predicted_classes = json.loads(predictions.json())

for i in range(len(x_new)):
    print ("Patient {}".format(x_new[i]), predicted_classes[i] )