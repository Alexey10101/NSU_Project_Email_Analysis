"""
загружает PASTED, SpamAssassin и GenImage в папку datasets проекта.
сохраняет исходные архивы, сведения об источниках и подготовленные данные по отдельным папкам.
для GenImage по умолчанию сохраняет небольшую выборку Tiny-GenImage; полный набор включается отдельно.
повторный запуск использует готовые файлы и продолжает незавершённые загрузки.
"""

import argparse
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from email import policy
from email.message import EmailMessage
from pathlib import Path, PurePosixPath


PASTED_REPO = "linzw/PASTED"
GENIMAGE_REPO = "ENSTA-U2IS/GenImage"
GENIMAGE_HOME = "https://github.com/GenImage-Dataset/GenImage"
GENIMAGE_NAMES = ("ADM", "BigGAN", "Midjourney", "VQDM", "glide",
                  "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "wukong")
SPAMASSASSIN_URL = "https://spamassassin.apache.org/old/publiccorpus/"
SPAMASSASSIN_FILES = {
    "easy_ham": "20030228_easy_ham.tar.bz2",
    "easy_ham_2": "20030228_easy_ham_2.tar.bz2",
    "hard_ham": "20030228_hard_ham.tar.bz2",
    "spam": "20030228_spam.tar.bz2",
    "spam_2": "20050311_spam_2.tar.bz2",
}
SEVEN_ZIP_URL = "https://github.com/ip7z/7zip/releases/download/26.05/"
SEVEN_ZIP_HASHES = {
    "7zr.exe": "9bf582b894dfa665f852d7e9112b02290a1c303c2d95a21fcdc41ffcb03143b7",
    "7z2605-extra.7z": "f27a8f103c84785026edcba1f67faf1357225df9d19ddb0b7760d55947141bc7",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def open_url(url, headers=None):
    request = urllib.request.Request(url, headers={"User-Agent": "NSU-Datasets/1.0", **(headers or {})})
    return urllib.request.urlopen(request, timeout=60)


def download(url, target, size=None, checksum=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = target.with_name(target.name + ".download.json")
    if target.is_file():
        receipt = read_json(receipt_path) if receipt_path.is_file() else {}
        info = target.stat()
        if (receipt.get("url") == url and receipt.get("size") == info.st_size
                and receipt.get("mtime_ns") == info.st_mtime_ns
                and (size is None or info.st_size == size)
                and (checksum is None or receipt.get("sha256") == checksum)):
            print(f"уже загружен: {target.name}", flush=True)
            return receipt["sha256"]
        if size is not None and info.st_size != size:
            raise ValueError(f"размер существующего файла не совпадает: {target}")
        digest = file_hash(target)
        if checksum is not None and digest != checksum:
            raise ValueError(f"контрольная сумма не совпадает: {target}; перемести файл и повтори запуск")
    else:
        partial = target.with_name(target.name + ".part")
        for attempt in range(3):
            try:
                offset = partial.stat().st_size if partial.is_file() else 0
                if size is not None and offset == size:
                    digest = file_hash(partial)
                    if checksum is not None and digest != checksum:
                        partial.unlink()
                        raise ValueError("контрольная сумма незавершённого файла не совпадает")
                    partial.replace(target)
                    break
                if size is not None and offset > size:
                    partial.unlink()
                    offset = 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                print(f"загрузка: {target.name}", flush=True)
                try:
                    response = open_url(url, headers)
                except urllib.error.HTTPError as error:
                    match = re.match(r"bytes \*/(\d+)", error.headers.get("Content-Range", ""))
                    if error.code != 416 or not match or offset != int(match[1]):
                        raise
                    digest = file_hash(partial)
                    if (size is not None and offset != size) or (checksum is not None and digest != checksum):
                        partial.unlink()
                        raise ValueError("незавершённый файл не совпадает с источником")
                    partial.replace(target)
                    break
                with response:
                    if response.status == 206:
                        match = re.match(r"bytes (\d+)-(\d+)/(\d+|\*)", response.headers.get("Content-Range", ""))
                        if not match or int(match[1]) != offset:
                            raise ValueError("сервер вернул неверную позицию продолжения загрузки")
                    else:
                        offset = 0
                    length = response.headers.get("Content-Length")
                    expected = offset + int(length) if length else size
                    mode = "ab" if offset else "wb"
                    written = offset
                    last_report = time.monotonic()
                    with partial.open(mode) as file:
                        while chunk := response.read(1024 * 1024):
                            file.write(chunk)
                            written += len(chunk)
                            if time.monotonic() - last_report >= 10:
                                print(f"{target.name}: {written / 2**20:.1f} МБ", flush=True)
                                last_report = time.monotonic()
                if (expected is not None and written != expected) or (size is not None and written != size):
                    raise OSError("загрузка оборвалась до получения полного файла")
                digest = file_hash(partial)
                if checksum is not None and digest != checksum:
                    partial.unlink()
                    raise ValueError("контрольная сумма скачанного файла не совпадает")
                partial.replace(target)
                break
            except (OSError, ValueError) as error:
                if attempt == 2:
                    raise
                print(f"повторная попытка: {error}", flush=True)
                time.sleep(attempt + 1)
        info = target.stat()
    save_json(receipt_path, dict(url=url, size=info.st_size, mtime_ns=info.st_mtime_ns, sha256=digest))
    return digest


def hf_files(repo, revision, directory=""):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}"
    if directory:
        url += "/" + urllib.parse.quote(directory, safe="/")
    url += "?limit=1000"
    files = []
    while url:
        with open_url(url) as response:
            files.extend(json.load(response))
            match = re.search(r'<([^>]+)>; rel="next"', response.headers.get("Link", ""))
            url = match[1] if match else None
    return files


def source_info(folder, repo, homepage):
    path = folder / "source.json"
    if path.is_file():
        return read_json(path)
    with open_url(f"https://huggingface.co/api/datasets/{repo}/revision/main") as response:
        revision = json.load(response)["sha"]
    result = dict(repo=repo, revision=revision, homepage=homepage)
    save_json(path, result)
    return result


def hf_download(info, remote, target):
    url = f"https://huggingface.co/datasets/{info['repo']}/resolve/{info['revision']}/"
    url += urllib.parse.quote(remote["path"], safe="/")
    return download(url, target, remote["size"], remote.get("lfs", {}).get("oid"))


def safe_path(root, name):
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
        raise ValueError(f"небезопасный путь в архиве: {name}")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"путь выходит за пределы папки: {name}")
    return target


def ready(folder, settings):
    marker = folder / "manifest.json"
    if not folder.exists():
        return False
    if not marker.is_file() or read_json(marker).get("settings") != settings:
        raise ValueError(f"в {folder} уже есть данные с другими параметрами; укажи другую --output-dir")
    print(f"подготовленные данные уже есть: {folder}", flush=True)
    return True


def clean_part(value):
    if not isinstance(value, str):
        raise ValueError("текстовое поле PASTED должно содержать строку")
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def make_body(row, edited):
    before = clean_part(row["precede_text"])
    middle = clean_part(row["para_text"] if edited else row["candidate_para_text"])
    after = clean_part(row["following_text"])
    if not middle:
        raise ValueError("пустой фрагмент в PASTED")
    start = len(before) + 2 if before else 0
    return "\n\n".join(part for part in (before, middle, after) if part), start, start + len(middle)


def make_eml(body, index, edited):
    msg = EmailMessage(policy=policy.default)
    msg["Subject"] = f"PASTED sample {index:06d}"
    msg["From"] = "sender@example.invalid"
    msg["To"] = "recipient@example.invalid"
    msg["Date"] = "Mon, 01 Jan 2024 00:00:00 +0000"
    msg["Message-ID"] = f"<pasted-{index:06d}-{'edited' if edited else 'human'}@example.invalid>"
    msg.set_content(body, charset="utf-8", cte="base64")
    return msg.as_bytes(policy=policy.SMTP)


def prepare_pasted(root, limit=None):
    folder = root / "PASTED"
    info = source_info(folder, PASTED_REPO, "https://github.com/Linzwcs/PASTED")
    settings = dict(version=1, revision=info["revision"], limit=limit)
    output = folder / "data"
    if ready(output, settings):
        return read_json(output / "manifest.json")
    remote = next(row for row in hf_files(info["repo"], info["revision"]) if row["path"] == "raw-GPT-3.5.zip")
    archive_path = folder / "raw" / remote["path"]
    digest = hf_download(info, remote, archive_path)
    records, assignments, indices = [], {}, set()
    order = {"train": 0, "validation": 1, "test": 2}
    with zipfile.ZipFile(archive_path) as archive:
        for source_split, member in (("train", "train.json"), ("validation", "dev.json"), ("test", "test.json")):
            count = 0
            with archive.open("raw-GPT-3.5/" + member) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if str(row["label"]) != "1":
                        continue
                    if not row["src"].endswith("_human"):
                        raise ValueError("метка PASTED противоречит источнику")
                    index = int(row["index"])
                    if index < 0 or index in indices:
                        raise ValueError(f"неверный или повторяющийся index: {index}")
                    indices.add(index)
                    original, _, _ = make_body(row, False)
                    normalized = re.sub(r"\s+", " ", original).strip().casefold()
                    group = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                    previous = assignments.get(group, "train")
                    assignments[group] = max((previous, source_split), key=order.get)
                    records.append((index, row, source_split, group))
                    count += 1
                    if limit is not None and count >= limit:
                        break
    counts = dict(train=0, validation=0, test=0)
    with tempfile.TemporaryDirectory(prefix=".pasted-build-", dir=folder) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        with (stage / "annotations.jsonl").open("w", encoding="utf-8") as annotations:
            for index, row, source_split, group in records:
                split = assignments[group]
                for edited, name in ((False, "human"), (True, "edited")):
                    body, start, end = make_body(row, edited)
                    relative = f"{split}/{name}/sample_{index:06d}.eml"
                    path = stage / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(make_eml(body, index, edited))
                    record = dict(file=relative, group_id=group, source_index=index, source=row["src"],
                                  split=split, source_split=source_split, ai_label=int(edited),
                                  generation_method="partial_paraphrase" if edited else "original",
                                  generator="GPT-3.5" if edited else None, language="en", email_type="synthetic_text_container",
                                  synthetic_headers=True, image_labels=[],
                                  generated_spans=[dict(zone="body", start=start, end=end)] if edited else [],
                                  offsets="символы text/plain после MIME-декодирования и strip; LF")
                    annotations.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
        result = dict(settings=settings, pairs=counts, archive_sha256=digest,
                      note="текстовый набор; заголовки искусственные, HTML и изображений нет")
        save_json(stage / "manifest.json", result)
        stage.rename(output)
    print(f"PASTED: подготовлено {sum(counts.values())} пар", flush=True)
    return result


def prepare_spamassassin(root):
    folder = root / "SpamAssassin"
    settings = dict(version=1, archives=SPAMASSASSIN_FILES)
    output = folder / "data"
    if ready(output, settings):
        return read_json(output / "manifest.json")
    archives = {}
    for category, filename in SPAMASSASSIN_FILES.items():
        path = folder / "raw" / filename
        archives[category] = (path, download(SPAMASSASSIN_URL + filename, path))
    counts = {}
    with tempfile.TemporaryDirectory(prefix=".spamassassin-build-", dir=folder) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        with (stage / "annotations.jsonl").open("w", encoding="utf-8") as annotations:
            for category, (path, _) in archives.items():
                counts[category] = 0
                with tarfile.open(path, "r:bz2") as archive:
                    for member in archive:
                        safe_path(stage, member.name)
                        if not member.isfile() or not re.fullmatch(r"\d+\.[a-fA-F0-9]+", PurePosixPath(member.name).name):
                            continue
                        name = PurePosixPath(member.name).name + ".eml"
                        target = stage / category / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.extractfile(member) as source, target.open("wb") as file:
                            shutil.copyfileobj(source, file)
                        record = dict(file=f"{category}/{name}", source="SpamAssassin", sha256=file_hash(target),
                                      spam_label=int(category.startswith("spam")), ai_label=None,
                                      note="разметка spam/ham не является разметкой происхождения ИИ")
                        annotations.write(json.dumps(record, ensure_ascii=False) + "\n")
                        counts[category] += 1
                if not counts[category]:
                    raise ValueError(f"в архиве {path.name} не найдены письма")
        result = dict(settings=settings, emails=counts, source=SPAMASSASSIN_URL,
                      archive_sha256={name: digest for name, (_, digest) in archives.items()},
                      ai_training_ready=False, note="нужны парные AI-версии и разметка изменений")
        save_json(stage / "manifest.json", result)
        stage.rename(output)
    save_json(folder / "source.json", dict(homepage=SPAMASSASSIN_URL, labels="spam/ham", ai_labels=False))
    print(f"SpamAssassin: подготовлено {sum(counts.values())} писем", flush=True)
    return result


def run_program(arguments):
    result = subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding="utf-8", errors="replace",
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        raise OSError(f"ошибка распаковки: {result.stdout[-2000:]}")
    return result.stdout


def seven_zip(root):
    for name in ("7zz", "7z", "7za"):
        if binary := shutil.which(name):
            return binary
    installed = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "7-Zip/7z.exe"
    if os.name == "nt" and installed.is_file():
        return str(installed)
    tools = root / ".tools" / "7zip"
    binary = tools / "x64/7za.exe"
    if os.name == "nt" and binary.is_file():
        return str(binary)
    if os.name != "nt" or platform.machine().lower() not in ("amd64", "x86_64"):
        raise OSError("для многотомного GenImage установи 7-Zip (7z или 7zz в PATH)")
    for name, checksum in SEVEN_ZIP_HASHES.items():
        download(SEVEN_ZIP_URL + name, tools / name, checksum=checksum)
    run_program([str(tools / "7zr.exe"), "x", str(tools / "7z2605-extra.7z"), f"-o{tools}", "-y"])
    if not binary.is_file():
        raise OSError("не найден портативный 7-Zip после распаковки")
    return str(binary)


def extract_genimage(binary, archive, output):
    listing = run_program([binary, "l", "-slt", "-ba", "-sccUTF-8", "--", str(archive)])
    for block in re.split(r"\r?\n\s*\r?\n", listing):
        match = re.search(r"^Path = (.+)$", block, flags=re.MULTILINE)
        if match:
            safe_path(output, match[1].strip())
        if "Symbolic Link =" in block or "Hard Link =" in block or re.search(r"^Attributes =.*\bl[rwx-]{9}", block, re.MULTILINE):
            raise ValueError("ссылки в архиве GenImage не поддерживаются")
    run_program([binary, "x", "-y", "-sccUTF-8", str(archive), f"-o{output}"])


def prepare_genimage(root, generators, download_only=False):
    folder = root / "GenImage_full"
    info = source_info(folder, GENIMAGE_REPO, GENIMAGE_HOME)
    info["mirror_note"] = "зеркало ENSTA-U2IS; контрольные суммы взяты из LFS Hugging Face"
    save_json(folder / "source.json", info)
    catalog = []
    for generator in generators:
        files = [row for row in hf_files(info["repo"], info["revision"], generator)
                 if row["type"] == "file" and re.search(r"\.(zip|z\d+)$", row["path"], re.IGNORECASE)]
        if not files or sum(row["path"].endswith(".zip") for row in files) != 1:
            raise ValueError(f"неожиданная структура архивов GenImage: {generator}")
        catalog.extend(files)
    save_json(folder / "catalog.json", dict(revision=info["revision"], files=catalog))
    total = sum(row["size"] for row in catalog)
    missing = sum(row["size"] for row in catalog if not (folder / "raw" / row["path"]).is_file())
    prepared = all((folder / "data" / name / "manifest.json").is_file() for name in generators)
    needed = missing + (total * 1.5 if not download_only and not prepared else 0)
    print(f"GenImage: {', '.join(generators)}; архивы {total / 2**30:.1f} ГБ", flush=True)
    if needed > shutil.disk_usage(root).free:
        raise OSError(f"недостаточно места для GenImage: требуется ориентировочно {needed / 2**30:.1f} ГБ")
    binary = None if download_only or prepared else seven_zip(root)
    results = {}
    for generator in generators:
        files = [row for row in catalog if PurePosixPath(row["path"]).parts[0] == generator]
        settings = dict(version=1, revision=info["revision"], generator=generator)
        output = folder / "data" / generator
        if not download_only and ready(output, settings):
            results[generator] = read_json(output / "manifest.json")
            continue
        for row in sorted(files, key=lambda item: item["path"]):
            hf_download(info, row, safe_path(folder / "raw", row["path"]))
        if download_only:
            results[generator] = dict(status="archives_downloaded", files=len(files))
            continue
        archive = next(row for row in files if row["path"].endswith(".zip"))
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".genimage-build-", dir=output.parent) as temporary:
            stage = Path(temporary) / "data"
            stage.mkdir()
            print(f"распаковка GenImage {generator}...", flush=True)
            extract_genimage(binary, folder / "raw" / archive["path"], stage)
            counts = {"ai": 0, "nature": 0}
            for path in stage.rglob("*"):
                if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    for label in counts:
                        if label in path.relative_to(stage).parts[:-1]:
                            counts[label] += 1
            if not all(counts.values()):
                raise ValueError(f"в GenImage {generator} не найдены оба класса изображений")
            result = dict(settings=settings, images=counts, labels={"nature": 0, "ai": 1},
                          splits="сохранены исходные train/val; выделение validation/test и дедупликация выполняются до обучения",
                          archives=[dict(path=row["path"], sha256=row.get("lfs", {}).get("oid")) for row in files])
            save_json(stage / "manifest.json", result)
            stage.rename(output)
            results[generator] = result
    return results


def prepare_genimage_sample(root, generators, limit):
    folder = root / "GenImage"
    repo = "TheKernel01/Tiny-GenImage"
    with open_url(f"https://huggingface.co/api/datasets/{repo}/revision/main") as response:
        revision = json.load(response)["sha"]
    names = {"ADM": "ADM", "BigGAN": "BigGAN", "Midjourney": "Midjourney", "VQDM": "VQDM",
             "glide": "GLIDE", "stable_diffusion_v_1_4": "SD14", "stable_diffusion_v_1_5": "SD15", "wukong": "Wukong"}
    selected = [names[name] for name in generators]
    settings = dict(version=1, repo=repo, revision=revision, generators=selected, images_per_class=limit)
    output = folder / "data"
    if ready(output, settings):
        return read_json(output / "manifest.json")
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "source.json", dict(repo=repo, revision=revision,
              homepage=f"https://huggingface.co/datasets/{repo}", original=GENIMAGE_HOME,
              license="cc-by-nc-sa-4.0", mode="viewer_sample",
              note="небольшая выборка изображений через Dataset Viewer; пригодность для анализа артефактов требует проверки"))
    total_bytes = 0
    budget = 128 * 1024 ** 2
    counts = {}
    with tempfile.TemporaryDirectory(prefix=".genimage-sample-", dir=folder) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        with (stage / "annotations.jsonl").open("w", encoding="utf-8") as annotations:
            for split in ("train", "validation"):
                targets = {"Real": limit}
                for index, name in enumerate(selected):
                    targets[name] = limit // len(selected) + int(index < limit % len(selected))
                found = dict.fromkeys(targets, 0)
                for offset in range(0, 10000, 100):
                    query = urllib.parse.urlencode(dict(dataset=repo, config="default", split=split, offset=offset, length=100))
                    with open_url("https://datasets-server.huggingface.co/rows?" + query) as response:
                        page = json.load(response)
                    features = {row["name"]: row["type"] for row in page["features"]}
                    if features["label"]["names"] != ["real", "fake"]:
                        raise ValueError("изменилась разметка Tiny-GenImage")
                    generator_names = features["generator"]["names"]
                    for item in page["rows"]:
                        row = item["row"]
                        generator = generator_names[row["generator"]]
                        if generator not in targets or found[generator] >= targets[generator]:
                            continue
                        label = int(row["label"])
                        if label != int(generator != "Real"):
                            raise ValueError("противоречивая разметка изображения")
                        source = row["image"]["src"]
                        if f"/--/{revision}/--/" not in source:
                            raise ValueError("версия Dataset Viewer отличается от текущей версии источника")
                        remaining = min(4 * 1024 ** 2, budget - total_bytes)
                        if remaining <= 0:
                            raise ValueError("выборка превысила ограничение 128 МиБ; уменьши --image-limit")
                        with open_url(source) as response:
                            data = response.read(remaining + 1)
                        if len(data) > remaining:
                            raise ValueError("изображение превышает ограничение размера выборки")
                        extension = ".png" if data.startswith(b"\x89PNG\r\n\x1a\n") else ".jpg" if data.startswith(b"\xff\xd8\xff") else None
                        if extension is None:
                            raise ValueError("сервер вернул неизвестный формат изображения")
                        relative = f"{split}/{'ai' if label else 'nature'}/{generator}_{item['row_idx']:06d}{extension}"
                        path = stage / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
                        total_bytes += len(data)
                        record = dict(file=relative, ai_label=label, generator=generator, split=split,
                                      source_repo=repo, source_revision=revision, source_row=item["row_idx"],
                                      sha256=hashlib.sha256(data).hexdigest(), width=row["image"]["width"],
                                      height=row["image"]["height"], source_format="dataset_viewer_asset",
                                      note="отдельное изображение, не вложение из письма")
                        annotations.write(json.dumps(record, ensure_ascii=False) + "\n")
                        found[generator] += 1
                    print(f"GenImage выборка {split}: {sum(found.values())}/{sum(targets.values())}", flush=True)
                    if all(found[name] == count for name, count in targets.items()):
                        break
                    if not page["rows"] or offset + 100 >= page["num_rows_total"]:
                        break
                if found != targets:
                    raise ValueError(f"не удалось собрать классы Tiny-GenImage: {found}, нужно {targets}")
                counts[split] = found
        result = dict(settings=settings, images=counts, bytes=total_bytes, mode="sample",
                      note="выборка для разработки; независимый test и дедупликация нужны перед обучением")
        save_json(stage / "manifest.json", result)
        stage.rename(output)
    print(f"GenImage: сохранена выборка, {total_bytes / 1024 ** 2:.1f} МиБ", flush=True)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "datasets")
    cli.add_argument("--datasets", nargs="+", choices=("pasted", "spamassassin", "genimage"),
                     default=["pasted", "spamassassin", "genimage"])
    cli.add_argument("--generators", nargs="+", choices=(*GENIMAGE_NAMES, "all"), default=["ADM", "BigGAN"])
    cli.add_argument("--limit", type=int, help="ограничение числа пар PASTED в каждом исходном разбиении")
    cli.add_argument("--image-limit", type=int, default=50, help="изображений каждого класса на разбиение")
    cli.add_argument("--full-genimage", action="store_true", help="скачать полные многогигабайтные архивы")
    cli.add_argument("--download-only", action="store_true", help="сохранить архивы GenImage без распаковки")
    args = cli.parse_args()
    if args.limit is not None and args.limit < 1:
        cli.error("--limit должен быть положительным числом")
    if args.image_limit < 1:
        cli.error("--image-limit должен быть положительным числом")
    if args.download_only and not args.full_genimage:
        cli.error("--download-only используется вместе с --full-genimage")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "manifest.json"
    state = read_json(state_path) if state_path.is_file() else {}
    state["pending"] = ["генерация локальной моделью полных писем, парных HTML-версий и их разметка",
                        "почтовые изображения, русские письма и сквозная проверка",
                        "проверка близких текстовых дубликатов перед обучением",
                        "дедупликация изображений и разделение GenImage перед обучением"]
    generators = list(GENIMAGE_NAMES) if "all" in args.generators else list(dict.fromkeys(args.generators))
    print(f"папка датасетов: {root}", flush=True)
    failed = False
    for name in dict.fromkeys(args.datasets):
        try:
            if name == "pasted":
                result = prepare_pasted(root, args.limit)
            elif name == "spamassassin":
                result = prepare_spamassassin(root)
            else:
                result = (prepare_genimage(root, generators, args.download_only) if args.full_genimage
                          else prepare_genimage_sample(root, generators, args.image_limit))
            state[name] = dict(status="downloaded" if name == "genimage" and args.download_only else "prepared", details=result)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, tarfile.TarError, StopIteration) as error:
            failed = True
            state[name] = dict(status="error", error=str(error))
            print(f"ошибка {name}: {error}", file=sys.stderr, flush=True)
        save_json(state_path, state)
    print(f"сводка сохранена: {state_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
