"""
Разбирает письма из файлов и байтов, извлекает текст, html и изображения.
Сохраняет заголовки и ссылки, читает каталоги писем и отдельные изображения.
Очищает текст от цитат, истории переписки и мобильных подписей.
"""

import re
import base64
import mimetypes
import sys
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from email.errors import MessageError
from email.utils import getaddresses
from pathlib import Path
from typing import Dict, Iterator, List
from urllib.parse import unquote_to_bytes
from bs4 import BeautifulSoup


@dataclass
class ExtractedImage:
    filename: str
    content_type: str
    data: bytes
    content_id: str = ""
    disposition: str = ""


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
    headers: Dict[str, List[str]] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)
    image_urls: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source_file: str = ""


class EmailParser:
    def __init__(self):
        self.policy = policy.default

    def parse_from_file(self, file_path: str) -> ParsedEmail:
        path = Path(file_path)
        parsed = self.parse_from_bytes(path.read_bytes())
        parsed.source_file = str(path.resolve())
        return parsed

    def parse_from_bytes(self, raw_bytes: bytes) -> ParsedEmail:
        if not raw_bytes.strip():
            raise ValueError("письмо пустое")
        msg = BytesParser(policy=self.policy).parsebytes(raw_bytes)
        return self._process_message(msg)

    def parse_from_directory(self, directory: str) -> Iterator[ParsedEmail]:
        folder = Path(directory)
        if not folder.is_dir():
            raise NotADirectoryError(f"не найдена папка: {folder}")
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() != ".eml":
                continue
            try:
                yield self.parse_from_file(str(path))
            except (OSError, ValueError, TypeError, MessageError) as error:
                print(f"не удалось разобрать {path}: {error}", file=sys.stderr)

    def _message_parts(self, msg):
        if msg.get_content_type() == "message/rfc822":
            return
        if msg.is_multipart():
            if msg.get_content_disposition() != "attachment":
                for part in msg.iter_parts():
                    yield from self._message_parts(part)
        else:
            yield msg

    def _decode_text(self, part, warnings):
        data = part.get_payload(decode=True)
        if data is None:
            payload = part.get_payload()
            return payload if isinstance(payload, str) else ""
        charset = part.get_content_charset() or "utf-8"
        for encoding in dict.fromkeys((charset, "utf-8", "cp1251", "latin-1")):
            try:
                text = data.decode(encoding)
                if encoding != charset:
                    warnings.append(f"вместо кодировки {charset} использована {encoding}")
                return text
            except (LookupError, UnicodeError):
                continue
        return data.decode("utf-8", errors="replace")

    def _process_message(self, msg) -> ParsedEmail:
        plain_texts = []
        html_texts = []
        images = []
        warnings = []

        # вложенные письма и текстовые вложения не добавляем в тело
        for part in self._message_parts(msg):
            content_type = part.get_content_type()

            if content_type.startswith("image/"):
                img_data = part.get_payload(decode=True)
                filename = part.get_filename() or "image.png"
                if img_data:
                    images.append(ExtractedImage(
                        filename=filename,
                        content_type=content_type,
                        data=img_data,
                        content_id=str(part.get("Content-ID", "")).strip("<>"),
                        disposition=part.get_content_disposition() or ""
                    ))

            elif content_type == "text/plain" and part.get_content_disposition() != "attachment":
                try:
                    plain_texts.append(self._decode_text(part, warnings))
                except (LookupError, UnicodeError, ValueError) as error:
                    warnings.append(f"не удалось прочитать текстовую часть: {error}")

            elif content_type == "text/html" and part.get_content_disposition() != "attachment":
                try:
                    html_texts.append(self._decode_text(part, warnings))
                except (LookupError, UnicodeError, ValueError) as error:
                    warnings.append(f"не удалось прочитать html-часть: {error}")

        full_plain = "\n".join(plain_texts).strip()
        full_html = "\n".join(html_texts).strip()

        # извлекаем текст из html, если он есть
        clean_from_html = ""
        links = []
        image_urls = []
        if full_html:
            soup = BeautifulSoup(full_html, "html.parser")
            links = [str(tag["href"]).strip() for tag in soup.find_all("a", href=True)]
            seen_images = set()
            for tag in soup.find_all("img", src=True):
                source = str(tag["src"]).strip()
                if source in seen_images:
                    continue
                seen_images.add(source)
                if source.lower().startswith("data:image/"):
                    try:
                        description, value = source.split(",", 1)
                        content_type = description[5:].split(";", 1)[0].lower()
                        data = unquote_to_bytes(value)
                        if ";base64" in description.lower():
                            data = base64.b64decode(re.sub(rb"\s+", b"", data), validate=True)
                        extension = mimetypes.guess_extension(content_type) or ".img"
                        if data:
                            images.append(ExtractedImage(f"inline_{len(images) + 1}{extension}",
                                                         content_type, data, disposition="inline"))
                    except ValueError as error:
                        warnings.append(f"не удалось прочитать встроенное изображение: {error}")
                elif source:
                    image_urls.append(source)
            # исходный html сохраняем, служебный код убираем только из текстовой копии
            for tag in soup(["script", "style", "head"]):
                tag.decompose()
            clean_from_html = soup.get_text(separator="\n", strip=True)
        links.extend(url.rstrip(".,;!?)>") for url in re.findall(r"https?://[^\s<>\"']+", full_plain))
        for part in msg.walk():
            warnings.extend(str(error) or type(error).__name__ for error in part.defects)

        return ParsedEmail(
            subject=str(msg.get("Subject", "")),
            sender=str(msg.get("From", "")),
            to=[address for _, address in getaddresses([str(value) for value in msg.get_all("To", [])]) if address],
            date=str(msg.get("Date", "")),
            text_plain=full_plain,
            text_html=full_html,
            extracted_text_from_html=clean_from_html,
            images=images,
            headers={name.lower(): [str(value) for value in msg.get_all(name)] for name in msg.keys()},
            links=list(dict.fromkeys(link for link in links if link)),
            image_urls=list(dict.fromkeys(image_urls)),
            warnings=list(dict.fromkeys(warnings))
        )


def load_image_from_file(file_path: str) -> ExtractedImage:
    path = Path(file_path)
    content_type = mimetypes.guess_type(str(path))[0]
    if not content_type or not content_type.startswith("image/"):
        raise ValueError(f"неподдерживаемый формат изображения: {path.name}")
    data = path.read_bytes()
    if not data:
        raise ValueError(f"изображение пустое: {path}")
    return ExtractedImage(path.name, content_type, data)


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
