import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Sample intent and answer data
data = {
    "intents": [
        {
            "intent": "install",
            "questions": [
                "how to install the software",
                "please install the software",
                "do you have a document to install the software"
            ]
        }
    ],
    "answers": [
        {
            "prediction": "install",
            "answer_text": "In order to install the software, please install Windows 11. After that, go to the C Drive, click on Program Files, and navigate to the installation directory."
        }
    ]
}

# Create a DataFrame for intents and answers
intents_df = pd.DataFrame(data["intents"])
answers_df = pd.DataFrame(data["answers"])

# Create and train a text classification model
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(intents_df['questions'].apply(lambda x: ' '.join(x)), intents_df['intent'])

# Save the trained model to a file
joblib.dump(model, 'intent_model.joblib')

# Function to get the intent based on user input using the trained model
def get_intent(user_input):
    predicted_intent = model.predict([user_input])
    return predicted_intent[0] if predicted_intent else "default"

# Function to generate response based on intent
def generate_response(intent):
    response_row = answers_df[answers_df["prediction"] == intent]
    if not response_row.empty:
        return response_row.iloc[0]["answer_text"]
    else:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"

# Main chat loop
while True:
    user_input = input("You: ")
    
    # Exit the loop if the user types 'exit'
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    
    # Get intent using the trained model and generate response
    intent = get_intent(user_input)
    response = generate_response(intent)
    
    # Print the chatbot's response
    print("Chatbot:", response)
