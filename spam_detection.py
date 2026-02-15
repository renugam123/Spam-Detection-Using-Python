import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 1: Dataset
data = {
    'message': [
        'Win a free iPhone now',
        'Call me later',
        'Congratulations you won a prize',
        'Hey how are you',
        'Claim your free reward',
        'Let us meet tomorrow',
        'Free entry in a contest',
        'Are we meeting today',
        'You have won cash prize',
        'Please call me'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)

# Step 2: Text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])

y = df['label']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 4: Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 5: Accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Step 6: User input prediction
while True:
    user_message = input("\nEnter a message to check (or type 'exit' to stop): ")

    if user_message.lower() == 'exit':
        print("Exiting program...")
        break

    message_vector = vectorizer.transform([user_message])
    prediction = model.predict(message_vector)

    print("Prediction:", prediction[0].upper())
