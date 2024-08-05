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
import chardet

# Specify the path to local NLTK data
nltk.data.path.append('D:\\CCA final project')

# Download necessary NLTK data
# Note: This line might be skipped if NLTK data is pre-included
# nltk.download('punkt')

# Path to files
file_path = "D:\\CCA final project\\spam.csv"
pickle_path = "D:\\CCA final project\\training_data.pkl"

# Detect file encoding
with open(file_path, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
encoding = result['encoding']

# Read the dataset file
df = pd.read_csv(file_path, encoding=encoding)
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
            open(pickle_path, "wb"))

# Load classifier and message data
datafile = pickle.load(open(pickle_path, "rb"))
message_x = datafile["message_x"]
classifier = datafile["classifier"]
tfvec = datafile["tfvec"]

# Function to connect to Gmail and fetch emails
def fetch_emails(limit=50):
    username = st.secrets["email"]
    password = st.secrets["password"]

    # Connect to the IMAP server
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(username, password)
    mail.select("inbox")  # Select inbox or another folder

    email_texts = {}
    try:
        # Fetch recent emails
        result, data = mail.search(None, "ALL")  # Fetch all emails
        email_ids = data[0].split()
        email_ids = email_ids[-limit:]  # Limit to the last `limit` emails
        for num in email_ids:
            result, data = mail.fetch(num, "(RFC822)")
            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)

            # Extract email content (subject and body)
            subject = msg["subject"]
            body = ""

            # Process each part of the message
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))

                if content_type == "text/plain" and "attachment" not in content_disposition:
                    # Decode text parts
                    try:
                        payload = part.get_payload(decode=True)
                        if isinstance(payload, bytes):
                            body += payload.decode('utf-8', 'ignore')
                        else:
                            body += payload
                    except Exception as e:
                        print(f"Error decoding message: {e}")
                        continue

            if body:
                email_texts[subject] = body
            else:
                email_texts[subject] = None

    except imaplib.IMAP4.error as e:
        print(f"IMAP error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return email_texts

def preprocess_message(message):
    # Preprocess the message
    msg = ''.join(filter(lambda ch: ch.isalpha() or ch == " ", message))
    words = word_tokenize(msg)
    stemmed_message = ' '.join([lstem.stem(word) for word in words])
    return stemmed_message

# Streamlit app layout
st.title('Spam Email Detector')

# Fetch emails button
if st.button('Fetch Emails'):
    with st.spinner('Fetching emails...'):
        email_texts = fetch_emails()
        if email_texts:
            st.session_state['emails'] = email_texts
            st.success('Emails fetched successfully!')
        else:
            st.error('No emails fetched.')

# Display emails in a select box
if 'emails' in st.session_state:
    emails = st.session_state['emails']
    subjects = list(emails.keys())
    if subjects:
        selected_subject = st.selectbox('Select an email to classify:', subjects)

        if selected_subject:
            email_body = emails[selected_subject]
            st.write(f"**Subject:** {selected_subject}")
            st.write(f"**Body:**\n{email_body}")

            # Classify button
            if st.button('Classify'):
                if email_body:
                    processed_msg = preprocess_message(email_body)
                    vectorized_msg = tfvec.transform([processed_msg]).toarray()

                    # Predict the label
                    prediction = classifier.predict(vectorized_msg)[0]
                    result = "spam" if prediction == 1 else "ham"
                    st.write(f"**Classification Result:** {result}")
                else:
                    st.write("Could not extract body.")
    else:
        st.write("No subjects available to display.")
else:
    st.write("No emails to display. Please fetch emails first.")
