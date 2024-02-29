# Intent-Based Chatbot

- **Objective**: Implement a simple intent-based chatbot using scikit-learn's CountVectorizer and Multinomial Naive Bayes. The chatbot is designed to understand user input and provide predefined responses based on identified intents.

- **Data**: The script uses a sample dataset containing intents and corresponding answers.

- **Model Training**: It creates a DataFrame from intent and answer data, then trains a text classification model using scikit-learn's make_pipeline.

- **Model Saving**: The trained model is saved to a file using joblib.

- **Chat Loop**: The script enters a chat loop where it continuously takes user input, predicts the intent using the trained model, and generates responses based on predefined answers.

- **Exit**: Type 'exit' to end the chat.
