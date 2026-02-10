from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import pickle

# Scopes define what your app can do
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

# Start OAuth flow
flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
creds = flow.run_local_server(port=0)

# Save the credentials for future use
with open('token.pkl', 'wb') as token_file:
    pickle.dump(creds, token_file)

# Build Gmail service
service = build('gmail', 'v1', credentials=creds)
print("Authenticated! Ready to send emails.")
