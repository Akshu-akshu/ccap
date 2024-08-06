import chardet
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split as ttsplit
from sklearn import svm
import pandas as pd
import pickle
import numpy as np
import imaplib
import email

# Detect file encoding
file = "spam.csv"
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
encoding = result['encoding']

# Read the dataset file
df = pd.read_csv(file, encoding=encoding)
message_X = df.iloc[:, 1]  # EmailText column
labels_Y = df.iloc[:, 0]  # Label

# Initialize the stemmer
lstem = LancasterStemmer()

def preprocess(messages):
    processed_messages = []
    for msg in messages:
        # Filter out non-alphabetic characters
        msg = ''.join(filter(lambda ch: ch.isalpha() or ch == " ", msg))
        # Tokenize the messages
        words = word_tokenize(msg)
        # Stem the words
        processed_messages.append(' '.join([lstem.stem(word) for word in words]))
    return processed_messages

message_x = preprocess(message_X)

# Vectorization process
tfvec = TfidfVectorizer(stop_words='english')
x_new = tfvec.fit_transform(message_x).toarray()

# Replace ham and spam labels with 0 and 1 respectively
y_new = np.array(labels_Y.replace(to_replace=['ham', 'spam'], value=[0, 1]))

# Split dataset into training and testing parts
x_train, x_test, y_train, y_test = ttsplit(x_new, y_new, test_size=0.2, shuffle=True)

# Train the SVM classifier
classifier = svm.SVC()
classifier.fit(x_train, y_train)

# Store the classifier and message features for prediction
pickle.dump({'classifier': classifier, 'message_x': message_x, 'tfvec': tfvec},
            open("training_data.pkl", "wb"))

# Load classifier and message data
datafile = pickle.load(open("training_data.pkl", "rb"))
classifier = datafile["classifier"]
tfvec = datafile["tfvec"]

# Function to connect to Gmail IMAP server
def connect_to_email():
    username = st.secrets["email"]  # Use your own email address
    password = st.secrets["password"]  # Use your own password

    # Connect to the IMAP server
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(username, password)
    mail.select("inbox")  # Select inbox or another folder

    return mail

# Function to fetch emails
def fetch_emails(mail, limit=50):
    email_texts = {}
    try:
        result, data = mail.search(None, "ALL")  # Fetch all emails
        email_ids = data[0].split()
        email_ids = email_ids[-limit:]
        for num in email_ids:
            result, data = mail.fetch(num, "(RFC822)")
            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)

            subject = msg["subject"]
            body = ""

            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))

                if content_type == "text/plain" and "attachment" not in content_disposition:
                    try:
                        payload = part.get_payload(decode=True)
                        if isinstance(payload, bytes):
                            body += payload.decode('utf-8', 'ignore')
                        else:
                            body += payload
                    except Exception as e:
                        st.error(f"Error decoding message: {e}")

            email_texts[subject] = body if body else None

    except imaplib.IMAP4.error as e:
        st.error(f"IMAP error occurred: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

    return email_texts

def preprocess_message(message):
    msg = ''.join(filter(lambda ch: ch.isalpha() or ch == " ", message))
    words = word_tokenize(msg)
    return ' '.join([lstem.stem(word) for word in words])

# Function to classify email
def classify_email(body):
    processed_msg = preprocess_message(body)
    vectorized_msg = tfvec.transform([processed_msg]).toarray()
    prediction = classifier.predict(vectorized_msg)[0]
    result = "spam" if prediction == 1 else "ham"
    return result

# Streamlit UI
st.title("Spam Detector")

# Button to fetch emails
if st.button("Fetch Emails"):
    with st.spinner('Fetching emails...'):
        mail = connect_to_email()
        email_texts = fetch_emails(mail)
        if email_texts:
            st.session_state.email_texts = email_texts
            st.session_state.email_list = list(email_texts.keys())
            st.success("Emails fetched successfully.")
        else:
            st.write("No emails found.")

# Display and classify emails
if 'email_list' in st.session_state:
    selected_email_subject = st.selectbox("Select an email:", st.session_state.email_list)
    if selected_email_subject:
        email_body = st.session_state.email_texts[selected_email_subject]

        if st.button("Classify"):
            if email_body:
                result = classify_email(email_body)
                st.write(f"Email '{selected_email_subject}' is: {result}")
            else:
                st.write("Email content is empty, likely spam.")

# Evaluate accuracy
accuracy = classifier.score(x_test, y_test)
st.write(f"Accuracy of the model: {accuracy:.2%}")
