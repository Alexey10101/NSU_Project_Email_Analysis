"""
Загружает и сохраняет человеческие письма и их версии отредактированные ИИ (GPT-3.5)
в папки human и edited с одинаковыми именами парных файлов.
При повторном запуске проверяет соответствие имён в уже существующем датасете.
"""

import argparse
import io
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from contextlib import contextmanager
from email import policy
from email.message import EmailMessage
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_URL = "https://huggingface.co/datasets/linzw/PASTED/resolve/main/raw-GPT-3.5.zip"
TRAIN_MEMBER = "raw-GPT-3.5/train.json"
REQUIRED = {"index", "precede_text", "candidate_para_text", "following_text",
            "para_text", "label", "src"}


def json_rows(stream):
    first = stream.readline()
    if first.lstrip().startswith("["):
        yield from json.loads(first + stream.read())
    else:
        if first.strip():
            yield json.loads(first)
        for line in stream:
            if line.strip():
                yield json.loads(line)


@contextmanager
def open_rows(source):
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            with archive.open(TRAIN_MEMBER) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8-sig") as stream:
                    yield json_rows(stream)
    elif source.suffix.lower() in {".json", ".jsonl"}:
        with source.open(encoding="utf-8-sig") as stream:
            yield json_rows(stream)
    else:
        raise ValueError("Источник должен быть JSON, JSONL или официальным ZIP")


def clean_part(value):
    if not isinstance(value, str):
        raise ValueError("Текстовые поля должны содержать строки")
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def make_body(row, folder):
    field = "candidate_para_text" if folder == "human" else "para_text"
    parts = [clean_part(row[key]) for key in ("precede_text", field, "following_text")]
    if not parts[1]:
        raise ValueError("Пустой исходный или отредактированный фрагмент")
    return "\n\n".join(part for part in parts if part)


def make_eml(body, source_index, folder):
    msg = EmailMessage(policy=policy.default)
    msg["Subject"] = f"PASTED sample {source_index:06d}"
    msg["From"] = "sender@example.invalid"
    msg["To"] = "recipient@example.invalid"
    msg["Date"] = "Mon, 01 Jan 2024 00:00:00 +0000"
    msg["Message-ID"] = f"<pasted-{source_index:06d}-{folder}@example.invalid>"
    msg.set_content(body, subtype="plain", charset="utf-8", cte="base64")
    return msg.as_bytes(policy=policy.SMTP)


def download_archive(target):
    print("Загрузка исходного PASTED с Hugging Face...", flush=True)
    request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "PASTED-EML/2.0"})
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)
    with zipfile.ZipFile(target) as archive:
        archive.getinfo(TRAIN_MEMBER)


def convert(source, output_dir, limit=None):
    if output_dir.exists():
        raise FileExistsError(f"Папка уже существует: {output_dir}. Укажите другую --output-dir.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pasted-build-", dir=output_dir.parent))
    pairs = skipped = 0
    seen_indices = set()
    try:
        with tempfile.TemporaryDirectory(prefix="pasted-source-") as temp:
            if source is None:
                source = Path(temp) / "raw-GPT-3.5.zip"
                download_archive(source)
            for folder in ("human", "edited"):
                (staging / folder).mkdir()
            with open_rows(source) as rows:
                for row in rows:
                    if limit is not None and pairs >= limit:
                        break
                    if not isinstance(row, dict) or not REQUIRED.issubset(row):
                        raise ValueError("Запись не содержит обязательных полей PASTED")
                    label = str(row["label"])
                    if label not in {"0", "1"}:
                        raise ValueError(f"Неизвестная метка: {label}")
                    if label == "0":
                        skipped += 1
                        continue
                    if not isinstance(row["src"], str) or not row["src"].endswith("_human"):
                        raise ValueError("Метка человеческого текста противоречит источнику")
                    index_string = str(row["index"])
                    if not index_string.isascii() or not index_string.isdecimal():
                        raise ValueError(f"Неверный index: {index_string}")
                    index = int(index_string)
                    if index in seen_indices:
                        raise ValueError(f"Повторяющийся index: {index}")
                    seen_indices.add(index)
                    filename = f"sample_{index:06d}.eml"
                    for folder in ("human", "edited"):
                        body = make_body(row, folder)
                        (staging / folder / filename).write_bytes(make_eml(body, index, folder))
                    pairs += 1
                    if pairs % 1000 == 0:
                        print(f"Создано пар: {pairs}", flush=True)
            if not pairs:
                raise ValueError("В источнике нет подходящих человеческих текстов")
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(f"Готово: {pairs} пар, по {pairs} писем в human и edited.")
    print(f"Пропущено машинных оригиналов: {skipped}")
    print(f"Папка: {output_dir}")
    return pairs


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path, help="Локальный JSON/JSONL или исходный ZIP")
    cli.add_argument("--output-dir", type=Path, default=BASE_DIR / "GPT3.5_dataset")
    cli.add_argument("--limit", type=int, help="Количество пар для пробного запуска")
    args = cli.parse_args()
    if args.limit is not None and args.limit < 1:
        cli.error("--limit должен быть положительным числом")
    try:
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = BASE_DIR / output_dir
        output_dir = output_dir.resolve()
        if output_dir.exists() and args.source is None and args.limit is None:
            for folder in ("human", "edited"):
                if not (output_dir / folder).is_dir():
                    raise ValueError(f"В существующем датасете нет папки {folder}: {output_dir}")
            human_names = {p.name for p in (output_dir / "human").glob("*.eml") if p.is_file()}
            edited_names = {p.name for p in (output_dir / "edited").glob("*.eml") if p.is_file()}
            if not human_names or human_names != edited_names:
                raise ValueError("Датасет неполный: папки пусты или имена пар не совпадают. "
                                 "Для новой загрузки укажите другую --output-dir.")
            print(f"Датасет уже загружен: {output_dir}")
            print(f"Проверено соответствие имён: {len(human_names)} пар.")
            print("Для сравнения случайной пары запустите test.py.")
            return 0
        source = args.source
        if source is not None:
            if not source.is_absolute():
                source = BASE_DIR / source
            source = source.resolve(strict=True)
        convert(source, output_dir, args.limit)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
