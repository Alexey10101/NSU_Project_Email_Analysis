"""
считает признаки html, заголовков и ссылок, оценивает признаки редактирования письма ии.
использует простой логистический классификатор и объясняет вклад двух признаков.
при прямом запуске обучается на письмах с html и сохраняет модель и калибровку в json.
без html или обученной модели возвращает признаки и причину отсутствия оценки.
"""

import argparse
import ipaddress
import json
import math
import random
import re
import sys
from email.utils import parseaddr
from itertools import groupby
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from bs4 import BeautifulSoup, Comment
from parser import EmailParser
from text_analyze import collect_rows, dataset_files, fit_logistic, sigmoid


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "html_classifier.json"
CALIBRATION_PATH = BASE_DIR / "models" / "html_calibration.json"
FEATURE_NAMES = (
    "tags_count", "tables_count", "inline_style_ratio", "hidden_elements_count",
    "comments_count", "images_count", "links_count", "unique_domains_count",
    "ip_links_count", "tracking_links_count", "url_mismatch_count",
    "headers_count", "received_count", "has_mailer", "has_list_unsubscribe",
    "reply_to_mismatch",
)
FEATURE_LABELS = (
    "число html-тегов", "число таблиц", "доля встроенных стилей", "скрытые элементы",
    "html-комментарии", "изображения в html", "число ссылок", "разные домены",
    "ссылки на ip-адреса", "ссылки с параметрами отслеживания", "несовпадение адреса ссылки с текстом",
    "число заголовков", "число received", "наличие почтового клиента", "заголовок отписки",
    "разные домены отправителя и reply-to",
)


def get_domain(url: str) -> str:
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in ("http", "https", ""):
            return ""
        return (parsed.hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def calculate_html_features(soup) -> dict:
    tags = soup.find_all(True)
    hidden = 0
    for tag in tags:
        style = re.sub(r"\s+", "", str(tag.get("style", "")).lower())
        if tag.has_attr("hidden") or str(tag.get("aria-hidden", "")).lower() == "true" or re.search(
                r"(?:^|;)(?:display:none|visibility:hidden|opacity:0(?:\.0+)?|font-size:0(?:px|em|rem|pt|%)?)(?:!important)?(?:;|$)", style):
            hidden += 1
    return {
        "tags_count": len(tags),
        "tables_count": len(soup.find_all("table")),
        "inline_style_ratio": round(sum(tag.has_attr("style") for tag in tags) / len(tags), 3) if tags else 0.0,
        "hidden_elements_count": hidden,
        "comments_count": len(soup.find_all(string=lambda text: isinstance(text, Comment))),
        "images_count": len(soup.find_all("img")),
    }


def calculate_link_features(soup, plain_text: str) -> dict:
    links = [(str(tag.get("href", "")).strip(), tag.get_text(" ", strip=True)) for tag in soup.find_all("a")]
    if not links:
        links = [(url.rstrip(".,;!?)>"), "") for url in re.findall(r"https?://[^\s<>\"']+", plain_text)]
    domains = set()
    count = ip_count = tracking_count = mismatch_count = 0
    for url, text in links:
        domain = get_domain(url)
        if not domain:
            continue
        count += 1
        domains.add(domain)
        try:
            ipaddress.ip_address(domain)
            ip_count += 1
        except ValueError:
            pass
        query = parse_qsl(urlsplit(url).query, keep_blank_values=True)
        if any(key.lower().startswith("utm_") or key.lower() in ("gclid", "fbclid") for key, _ in query):
            tracking_count += 1
        visible_domain = get_domain(text)
        if visible_domain and visible_domain != domain:
            mismatch_count += 1
    return {
        "links_count": count,
        "unique_domains_count": len(domains),
        "ip_links_count": ip_count,
        "tracking_links_count": tracking_count,
        "url_mismatch_count": mismatch_count,
    }


def calculate_header_features(headers: dict) -> dict:
    sender = parseaddr(str(headers.get("from", [""])[0]))[1].lower().rpartition("@")[2]
    reply_to = parseaddr(str(headers.get("reply-to", [""])[0]))[1].lower().rpartition("@")[2]
    return {
        "headers_count": sum(len(values) for values in headers.values()),
        "received_count": len(headers.get("received", [])),
        "has_mailer": int("x-mailer" in headers or "user-agent" in headers),
        "has_list_unsubscribe": int("list-unsubscribe" in headers),
        "reply_to_mismatch": int(bool(sender and reply_to and sender != reply_to)),
    }


def feature_vector(features: dict) -> list:
    return [math.log1p(max(0.0, float(features[name]))) for name in FEATURE_NAMES]


class HtmlClassifier:
    def __init__(self, model: dict, calibration: dict):
        if model["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("список признаков изменился, нужно повторное обучение html_analyze.py")
        self.model = model
        self.calibration = calibration

    def transform(self, features: dict) -> list:
        return [(x - mean) / scale for x, mean, scale in
                zip(feature_vector(features), self.model["mean"], self.model["scale"])]

    def decision(self, features: dict) -> float:
        return self.model["bias"] + sum(w * x for w, x in zip(self.model["weights"], self.transform(features)))

    def probability(self, features: dict) -> float:
        return sigmoid(self.calibration["weight"] * self.decision(features) + self.calibration["bias"])

    def explain(self, features: dict) -> list:
        values = [w * x * self.calibration["weight"] for w, x in
                  zip(self.model["weights"], self.transform(features))]
        strongest = sorted(range(len(values)), key=lambda i: abs(values[i]), reverse=True)[:2]
        reasons = []
        for index in strongest:
            if abs(values[index]) < 0.01:
                continue
            direction = "повышает" if values[index] > 0 else "снижает"
            reasons.append(f"{FEATURE_LABELS[index]}: {features[FEATURE_NAMES[index]]}; {direction} оценку")
        return reasons or ["признаки близки к средним значениям обучающей выборки"]


def extract_all_features(html: str, headers: dict = None, plain_text: str = "") -> dict:
    soup = BeautifulSoup(html, "html.parser")
    return {
        **calculate_html_features(soup),
        **calculate_link_features(soup, plain_text),
        **calculate_header_features(headers or {}),
    }


def analyze_email_from_bytes(raw_bytes: bytes, include_score: bool = True) -> dict:
    parsed = EmailParser().parse_from_bytes(raw_bytes)
    html = parsed.text_html
    features = extract_all_features(html, parsed.headers, parsed.text_plain)
    result = dict(subject=parsed.subject, method="html_features_logistic",
                  has_html=bool(html), html_ai_index=None, html_ai_score=None, html_ai_verdict=None,
                  html_ai_status="not_evaluated" if html else "no_html",
                  html_ai_reasons=["оценка отключена"] if html else ["нет html-части, признаки заголовков и ссылок сохранены"],
                  **features)
    if not html or not include_score:
        return result
    if not MODEL_PATH.is_file() or not CALIBRATION_PATH.is_file():
        result.update(html_ai_status="not_trained", html_ai_reasons=["html-классификатор ещё не обучен"])
        return result
    classifier = HtmlClassifier(json.loads(MODEL_PATH.read_text(encoding="utf-8")),
                                json.loads(CALIBRATION_PATH.read_text(encoding="utf-8")))
    probability = classifier.probability(features)
    result.update(html_ai_index=round(probability, 6), html_ai_score=round(probability * 100.0, 1),
                  html_ai_verdict="AI-Generated" if probability >= classifier.calibration["threshold"] else "Human-Written",
                  html_ai_status="ok", html_ai_reasons=classifier.explain(features))
    return result


def analyze_email_from_file(file_path: str, include_score: bool = True) -> dict:
    return analyze_email_from_bytes(Path(file_path).read_bytes(), include_score)


def calculate_metrics(probabilities, labels, weights, threshold: float) -> dict:
    tp = fp = fn = 0.0
    for probability, label, weight in zip(probabilities, labels, weights):
        if probability >= threshold:
            if label == 1:
                tp += weight
            else:
                fp += weight
        elif label == 1:
            fn += weight
    positive = sum(w for y, w in zip(labels, weights) if y == 1)
    negative = sum(w for y, w in zip(labels, weights) if y == 0)
    lower_negative = wins = 0.0
    for _, tied in groupby(sorted(zip(probabilities, labels, weights)), key=lambda row: row[0]):
        rows = list(tied)
        tied_positive = sum(w for _, y, w in rows if y == 1)
        tied_negative = sum(w for _, y, w in rows if y == 0)
        wins += tied_positive * (lower_negative + 0.5 * tied_negative)
        lower_negative += tied_negative
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                roc_auc=wins / (positive * negative) if positive and negative else None,
                fpr=fp / negative if negative else 0.0)


def train_classifier(dataset_dir: Path = BASE_DIR / "datasets" / "PASTED" / "data") -> dict:
    print("чтение писем с html...", flush=True)
    groups = {}
    for path, label, group, _ in dataset_files(dataset_dir):
        values = analyze_email_from_file(str(path), include_score=False)
        if values["has_html"]:
            groups.setdefault(group, []).append((values, label))
    if len(groups) < 20 or any({y for _, y in rows} != {0, 1} for rows in groups.values()):
        raise ValueError("нужно хотя бы 20 полных групп с html в human и edited; обычный PASTED без html не подходит")
    names = sorted(groups)
    if (dataset_dir / "train").is_dir():
        train_names = [name for name in names if name.startswith("train:")]
        validation_names = [name for name in names if name.startswith("validation:")]
        test_names = [name for name in names if name.startswith("test:")]
    else:
        random.Random(42).shuffle(names)
        first, second = int(len(names) * 0.7), int(len(names) * 0.85)
        train_names, validation_names, test_names = names[:first], names[first:second], names[second:]
    if not all((train_names, validation_names, test_names)):
        raise ValueError("нужны непустые train, validation и test с html")
    train_x, train_y, train_w = collect_rows(groups, train_names)
    validation_x, validation_y, validation_w = collect_rows(groups, validation_names)
    test_x, test_y, test_w = collect_rows(groups, test_names)
    vectors = [feature_vector(values) for values in train_x]
    total = sum(train_w)
    mean = [sum(row[i] * w for row, w in zip(vectors, train_w)) / total for i in range(len(FEATURE_NAMES))]
    scale = [max(math.sqrt(sum(w * (row[i] - mean[i]) ** 2 for row, w in zip(vectors, train_w)) / total), 1e-8)
             for i in range(len(FEATURE_NAMES))]
    scaled = [[(x - m) / s for x, m, s in zip(row, mean, scale)] for row in vectors]
    print(f"обучение на {len(train_names)} группах...", flush=True)
    weights, bias = fit_logistic(scaled, train_y, train_w)
    model = dict(feature_names=list(FEATURE_NAMES), weights=weights, bias=bias, mean=mean, scale=scale, seed=42,
                 split_groups=dict(train=train_names, validation=validation_names, test=test_names))
    calibration = dict(weight=1.0, bias=0.0, threshold=0.5)
    classifier = HtmlClassifier(model, calibration)
    logits = [[classifier.decision(values)] for values in validation_x]
    fitted, bias = fit_logistic(logits, validation_y, validation_w, epochs=400, penalty=0.001)
    calibration.update(weight=fitted[0], bias=bias)
    validation_p = [classifier.probability(values) for values in validation_x]
    thresholds = sorted({0.0, 0.5, 1.0, *validation_p})
    candidates = [(t, calculate_metrics(validation_p, validation_y, validation_w, t)) for t in thresholds]
    calibration["threshold"] = max((row for row in candidates if row[1]["fpr"] <= 0.2),
                                   key=lambda row: row[1]["f1"])[0]
    probabilities = [classifier.probability(values) for values in test_x]
    metrics = calculate_metrics(probabilities, test_y, test_w, calibration["threshold"])
    model["metrics"] = metrics
    MODEL_PATH.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    CALIBRATION_PATH.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"модель сохранена: {MODEL_PATH}")
    print(f"калибровка сохранена: {CALIBRATION_PATH}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("file", nargs="?", type=Path, help="письмо для анализа; без аргумента выполняется обучение")
    cli.add_argument("--dataset", type=Path, default=BASE_DIR / "datasets" / "PASTED" / "data")
    args = cli.parse_args()
    try:
        if args.file is not None:
            print(json.dumps(analyze_email_from_file(str(args.file)), ensure_ascii=False, indent=2))
        else:
            train_classifier(args.dataset)
    except (OSError, ValueError, KeyError) as error:
        print(f"ошибка html-анализа: {error}", file=sys.stderr)
        sys.exit(1)
