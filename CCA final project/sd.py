import streamlit as st
import imaplib
import email
from email.header import decode_header
import chardet

def fetch_emails():
    try:
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        mail.login('talavarakshatha57@gmail.com', 'ugwx rprc mcou lmyg')  # Replace with your email and app password
        mail.select('inbox')

        status, data = mail.search(None, 'ALL')
        email_ids = data[0].split()

        emails = []
        for e_id in email_ids:
            status, msg_data = mail.fetch(e_id, '(RFC822)')
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject = decode_header(msg['subject'])[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode()
                    from_ = msg['from']
                    
                    # Handle email body
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == 'text/plain':
                                charset = part.get_content_charset() or 'utf-8'
                                try:
                                    body = part.get_payload(decode=True).decode(charset)
                                except (UnicodeDecodeError, TypeError):
                                    raw_data = part.get_payload(decode=True)
                                    detected_encoding = chardet.detect(raw_data)['encoding']
                                    try:
                                        body = raw_data.decode(detected_encoding)
                                    except (UnicodeDecodeError, TypeError):
                                        body = "Unable to decode email body"
                                break
                    else:
                        charset = msg.get_content_charset() or 'utf-8'
                        try:
                            body = msg.get_payload(decode=True).decode(charset)
                        except (UnicodeDecodeError, TypeError):
                            raw_data = msg.get_payload(decode=True)
                            detected_encoding = chardet.detect(raw_data)['encoding']
                            try:
                                body = raw_data.decode(detected_encoding)
                            except (UnicodeDecodeError, TypeError):
                                body = "Unable to decode email body"
                    
                    emails.append((subject, from_, body))

        mail.logout()
        return emails

    except Exception as e:
        print(f"Error fetching emails: {e}")
        return []

def main():
    st.title('Email Classifier')

    # Fetch emails
    emails = fetch_emails()
    
    st.write(f"Fetched {len(emails)} emails.")  # Debugging line

    if not emails:
        st.write("No emails found.")
        return

    # Create a list of email subjects
    email_subjects = [email[0] for email in emails]

    # Select an email
    selected_subject = st.selectbox('Select an email', email_subjects)

    if selected_subject:
        # Find the selected email
        selected_email = next(email for email in emails if email[0] == selected_subject)
        subject, from_, body = selected_email

        st.write(f"**From:** {from_}")
        st.write(f"**Subject:** {subject}")
        st.write(f"**Body:** {body}")

if __name__ == "__main__":
    main()
