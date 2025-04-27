import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Asha"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # Special handling for job listings which are stored as a list of objects
                if tag == "jobs":
                    # Format the first 5 job listings for display
                    job_listings = intent['responses'][0][:5]  # Get first 5 jobs
                    job_response = "Here are some job opportunities:\n\n"
                    
                    for job in job_listings:
                        job_response += f"🔹 {job['job_title']} at {job['company']}\n"
                        job_response += f"   Location: {job['location']} | Work Type: {job['work_type']}\n"
                        job_response += f"   Experience: {job['experience']} | Skills: {', '.join(job['skills'])}\n\n"
                    
                    job_response += "Type 'more jobs' to see additional listings."
                    return job_response
                else:
                    return random.choice(intent['responses'])
    
    return "I don't understand. Could you please rephrase your question?"


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

