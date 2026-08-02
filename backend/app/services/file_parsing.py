import email
from io import BytesIO

from pypdf import PdfReader


def extract_text_from_pdf(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_eml(data: bytes) -> str:
    message = email.message_from_bytes(data)
    parts = [f"Subject: {message.get('Subject', '')}", f"From: {message.get('From', '')}"]
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain":
                parts.append(part.get_payload(decode=True).decode(errors="replace"))
    else:
        parts.append(message.get_payload(decode=True).decode(errors="replace"))
    return "\n".join(parts)
