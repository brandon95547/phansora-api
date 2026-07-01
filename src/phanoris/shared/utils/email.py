import os
from email.message import EmailMessage
from aiosmtplib import send
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
EMAIL_TO = os.getenv("EMAIL_TO")      # 👈 env-controlled recipient
DEFAULT_FROM = os.getenv("DEFAULT_FROM", SMTP_USER)


async def send_email(data):
    # Always send to EMAIL_TO from env
    to_address = EMAIL_TO

    subject = data.get("subject", "No Subject")
    message_body = data.get("message", "")

    if not message_body:
        raise ValueError("Missing 'message'")

    msg = EmailMessage()
    msg["From"] = DEFAULT_FROM
    msg["To"] = to_address
    msg["Subject"] = subject
    msg.set_content(message_body)

    # Let SMTP/connection errors propagate so the caller can return a
    # non-2xx status instead of reporting a false success.
    await send(
        msg,
        hostname=SMTP_HOST,
        port=SMTP_PORT,
        username=SMTP_USER,
        password=SMTP_PASS,
        start_tls=True,
    )
    return "Email sent"
