import time
from twilio.rest import Client

# Twilio Configuration
# REPLACE THESE WITH YOUR ACTUAL TWILIO CREDENTIALS
TWILIO_ACCOUNT_SID = 'your_account_sid'
TWILIO_AUTH_TOKEN = 'your_auth_token'
TWILIO_PHONE_NUMBER = 'your_twilio_number'
DESTINATION_PHONE_NUMBER = 'your_destination_mobile_number'

COOLDOWN_PERIOD = 60 # seconds between alerts
last_alert_time = 0

def send_mobile_alert(message_body):
    global last_alert_time
    current_time = time.time()
    
    # Check cooldown
    if current_time - last_alert_time < COOLDOWN_PERIOD:
        return False, "Alert on cooldown."
        
    if TWILIO_ACCOUNT_SID == 'your_account_sid' or not TWILIO_ACCOUNT_SID:
        print("Warning: Twilio credentials not set. Mobile alert skipped.")
        return False, "Credentials not configured."
        
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=DESTINATION_PHONE_NUMBER
        )
        print(f"Alert sent to mobile successfully: SID {message.sid}")
        last_alert_time = current_time
        return True, "Alert sent successfully!"
    except Exception as e:
        print(f"Failed to send mobile alert: {e}")
        return False, f"Error: {e}"
