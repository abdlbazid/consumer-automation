"""
Gmail API Helper Module
Official Google Gmail API wrapper with OAuth2 authentication
"""

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import logging

logger = logging.getLogger(__name__)

# ---------------- Gmail API Setup ----------------
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def get_gmail_service():
    """Authenticate and return Gmail API service."""
    creds = None
    # token.json stores user's access and refresh tokens
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for next time
        with open('token.json', 'w') as token_file:
            token_file.write(creds.to_json())
        
        logger.info("Gmail credentials saved to token.json")
    
    service = build('gmail', 'v1', credentials=creds)
    return service

# ---------------- Email Creation Helpers ----------------
def create_message(to, subject, body, attachments=None):
    """Create a MIME email message with optional attachments."""
    if attachments is None:
        attachments = []

    if attachments:
        msg = MIMEMultipart()
        msg.attach(MIMEText(body, 'plain'))
        for path in attachments:
            part = MIMEBase('application', 'octet-stream')
            with open(path, 'rb') as f:
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(path)}"')
            msg.attach(part)
    else:
        msg = MIMEText(body, 'plain')

    msg['To'] = to
    msg['Subject'] = subject
    msg['From'] = 'me'
    
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode('utf-8')
    return {'raw': raw}

# ---------------- Email Operations ----------------
def send_email(to, subject, body, attachments=None):
    """Send an email immediately."""
    try:
        service = get_gmail_service()
        message = create_message(to, subject, body, attachments)
        result = service.users().messages().send(userId='me', body=message).execute()
        logger.info("Email sent to %s with messageId %s", to, result.get('id'))
        return result
    except Exception as e:
        logger.exception("Failed to send email: %s", e)
        raise

def create_draft(to, subject, body, attachments=None):
    """Create an email draft."""
    try:
        service = get_gmail_service()
        message = create_message(to, subject, body, attachments)
        draft = service.users().drafts().create(userId='me', body={'message': message}).execute()
        logger.info("Draft created with id %s", draft.get('id'))
        return draft
    except Exception as e:
        logger.exception("Failed to create draft: %s", e)
        raise

def send_draft(draft_id):
    """Send an existing draft by draft ID."""
    try:
        service = get_gmail_service()
        sent = service.users().drafts().send(userId='me', body={'id': draft_id}).execute()
        logger.info("Draft %s sent", draft_id)
        return sent
    except Exception as e:
        logger.exception("Failed to send draft: %s", e)
        raise

def list_messages(query=None, label_ids=None, max_results=50):
    """List message IDs matching optional query and labels."""
    try:
        service = get_gmail_service()
        results = service.users().messages().list(
            userId='me', 
            q=query, 
            labelIds=label_ids, 
            maxResults=max_results
        ).execute()
        messages = results.get('messages', [])
        logger.info("Found %d messages", len(messages))
        return messages
    except Exception as e:
        logger.exception("Failed to list messages: %s", e)
        raise

def get_message(message_id, format='full'):
    """Get a message by ID. format='full'|'metadata'|'minimal'|'raw'"""
    try:
        service = get_gmail_service()
        message = service.users().messages().get(userId='me', id=message_id, format=format).execute()
        logger.info("Retrieved message %s", message_id)
        return message
    except Exception as e:
        logger.exception("Failed to get message: %s", e)
        raise

def delete_message(message_id):
    """Delete a message permanently."""
    try:
        service = get_gmail_service()
        service.users().messages().delete(userId='me', id=message_id).execute()
        logger.info("Deleted message %s", message_id)
        return True
    except Exception as e:
        logger.exception("Failed to delete message: %s", e)
        raise

def modify_labels(message_id, add_labels=None, remove_labels=None):
    """Add or remove labels from a message."""
    try:
        service = get_gmail_service()
        body = {}
        if add_labels: 
            body['addLabelIds'] = add_labels
        if remove_labels: 
            body['removeLabelIds'] = remove_labels
        
        result = service.users().messages().modify(userId='me', id=message_id, body=body).execute()
        logger.info("Modified labels for message %s", message_id)
        return result
    except Exception as e:
        logger.exception("Failed to modify labels: %s", e)
        raise

def list_labels():
    """List all labels in the user's Gmail account."""
    try:
        service = get_gmail_service()
        results = service.users().labels().list(userId='me').execute()
        labels = results.get('labels', [])
        logger.info("Found %d labels", len(labels))
        return labels
    except Exception as e:
        logger.exception("Failed to list labels: %s", e)
        raise

def get_attachment(message_id, attachment_id, save_path):
    """Download an attachment to local file."""
    try:
        service = get_gmail_service()
        attachment = service.users().messages().attachments().get(
            userId='me', messageId=message_id, id=attachment_id
        ).execute()
        data = base64.urlsafe_b64decode(attachment['data'])
        with open(save_path, 'wb') as f:
            f.write(data)
        logger.info("Downloaded attachment to %s", save_path)
        return save_path
    except Exception as e:
        logger.exception("Failed to get attachment: %s", e)
        raise