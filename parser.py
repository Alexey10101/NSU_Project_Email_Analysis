"""
Разбирает письма из файлов и байтов, извлекает текст, html и изображения.
Очищает текст от цитат, истории переписки и мобильных подписей.
"""

import re
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from typing import List
from bs4 import BeautifulSoup


@dataclass
class ExtractedImage:
    filename: str
    content_type: str
    data: bytes


@dataclass
class ParsedEmail:
    subject: str
    sender: str
    to: List[str]
    date: str
    text_plain: str
    text_html: str
    extracted_text_from_html: str
    images: List[ExtractedImage] = field(default_factory=list)


class EmailParser:
    def __init__(self):
        self.policy = policy.default

    def parse_from_file(self, file_path: str) -> ParsedEmail:
        with open(file_path, "rb") as f:
            msg = BytesParser(policy=self.policy).parse(f)
        return self._process_message(msg)

    def parse_from_bytes(self, raw_bytes: bytes) -> ParsedEmail:
        msg = BytesParser(policy=self.policy).parsebytes(raw_bytes)
        return self._process_message(msg)

    def _process_message(self, msg) -> ParsedEmail:
        plain_texts = []
        html_texts = []
        images = []

        # пропускаем контейнеры и разбираем части письма
        for part in msg.walk():
            if part.is_multipart():
                continue

            content_type = part.get_content_type()

            if content_type.startswith("image/"):
                img_data = part.get_payload(decode=True)
                filename = part.get_filename() or "image.png"
                if img_data:
                    images.append(ExtractedImage(
                        filename=filename,
                        content_type=content_type,
                        data=img_data
                    ))

            elif content_type == "text/plain":
                try:
                    plain_texts.append(part.get_content())
                except Exception:
                    pass

            elif content_type == "text/html":
                try:
                    html_texts.append(part.get_content())
                except Exception:
                    pass

        full_plain = "\n".join(plain_texts).strip()
        full_html = "\n".join(html_texts).strip()

        # извлекаем текст из html, если он есть
        clean_from_html = ""
        if full_html:
            soup = BeautifulSoup(full_html, "html.parser")
            clean_from_html = soup.get_text(separator="\n", strip=True)

        return ParsedEmail(
            subject=str(msg.get("Subject", "")),
            sender=str(msg.get("From", "")),
            to=[str(msg.get("To", ""))],
            date=str(msg.get("Date", "")),
            text_plain=full_plain,
            text_html=full_html,
            extracted_text_from_html=clean_from_html,
            images=images
        )


def clean_email_body(text: str) -> str:
    if not text:
        return ""

    # отрезаем историю переписки по распространённым разделителям
    history_patterns = [
        r"-----Original Message-----",
        r"-----Исходное сообщение-----",
        r"_{5,}",  # разделитель из подчёркиваний
        r"On .* wrote:",
        r"\d{1,2} .* написал(\(а\))?:",
    ]
    for pattern in history_patterns:
        parts = re.split(pattern, text, flags=re.IGNORECASE)
        text = parts[0]

    clean_lines = []
    for line in text.splitlines():
        line_stripped = line.strip()

        # пропускаем цитаты и мобильные подписи
        if line_stripped.startswith(">"):
            continue

        if re.search(r"sent from my|отправлено с (моего )?(iphone|android|galaxy)", line_stripped, re.IGNORECASE):
            continue

        clean_lines.append(line)

    # оставляем не больше одной пустой строки подряд
    result = "\n".join(clean_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()
