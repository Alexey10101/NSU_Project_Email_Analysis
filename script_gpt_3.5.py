import os
import re
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

from datasets import load_dataset


ARCHIVE_URL = (
    "https://huggingface.co/datasets/linzw/PASTED/"
    "resolve/main/raw-GPT-3.5.zip"
)

TRAIN_FILE = (
    f"zip://raw-GPT-3.5/train.json::{ARCHIVE_URL}"
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def sanitize_filename(name: str, max_len: int = 80) -> str:
    # убираем недопустимые символы из имени файла
    name = re.sub(r"[^\w\-.]+", "_", str(name))
    return name[:max_len] or "sample"


def row_to_eml(row, idx: int, output_dir: str) -> str:
    # формируем .eml файл из одной строки датасета
    msg = EmailMessage()

    src = str(row.get("src", "unknown"))
    label = str(row.get("label", "unknown"))

    # тема письма это источник и метка
    msg["Subject"] = f"PASTED sample #{idx + 1} | src={src} | label={label}"
    msg["From"] = "pasted-dataset@local"
    msg["To"] = "recipient@local"
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="pasted.local")
    msg["X-Sample-Index"] = str(idx + 1)
    msg["X-Source"] = src
    msg["X-Label"] = label

    # тело письма это обычный текст с исходными полями
    body = (
        f"предшествующий текст:\n{row.get('precede_text', '')}\n\n"
        f"кандидат на перефразирование:\n{row.get('candidate_para_text', '')}\n\n"
        f"перефразированный текст:\n{row.get('para_text', '')}\n\n"
        f"следующий текст:\n{row.get('following_text', '')}\n\n"
        f"источник: {src}\n"
        f"метка: {label}\n"
    )
    msg.set_content(body)

    filename = f"sample_{idx + 1:05d}_{sanitize_filename(src)}.eml"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "wb") as f:
        f.write(msg.as_bytes())

    return filepath


def main():
    print("=== загрузка raw-GPT-3.5 из датасета PASTED ===")

    try:
        train_dataset = load_dataset(
            "json",
            data_files={"train": TRAIN_FILE},
            split="train",
        )
    except Exception as e:
        print(f"ошибка при загрузке датасета: {e}")
        return

    df = train_dataset.to_pandas()

    print(f"\n[+] загружено записей: {len(df)}")
    print(f"[+] колонки: {list(df.columns)}")

    # сохраняем .eml рядом со скриптом
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    saved = 0
    for idx, row in df.iterrows():
        try:
            row_to_eml(row, idx, OUTPUT_DIR)
            saved += 1
        except Exception as e:
            print(f"[!] ошибка при сохранении записи #{idx + 1}: {e}")

    print(f"\n[+] сохранено .eml файлов: {saved}")
    print(f"[+] папка: {OUTPUT_DIR}")

    # оставимч и csv для удобного анализа
    csv_path = os.path.join(OUTPUT_DIR, "raw_gpt35_dataset.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[+] также сохранён csv: {csv_path}")


if __name__ == "__main__":
    main()