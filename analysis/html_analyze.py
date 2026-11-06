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
import re
import sys
from email.utils import parseaddr
from email.errors import MessageError
import hashlib
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from bs4 import BeautifulSoup, Comment
from parser import EmailParser
from text_analyze import fit_logistic, sigmoid
from model_utils import analyze_path, calculate_metrics, calibrate, print_error, print_json, read_dataset, save_json, split_summary


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "html_classifier.json"
CALIBRATION_PATH = BASE_DIR / "models" / "html_calibration.json"
DATASET_PATH = BASE_DIR / "datasets" / "HTML_AI" / "data"
REPORT_PATH = BASE_DIR / "training" / "reports" / "html_metrics.json"
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
        url = url.strip()
        if "://" not in url and not url.startswith("//") and re.match(r"(?:www\.|[\w-]+\.[\w-]+)", url):
            url = "https://" + url
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
    html_urls = {url for url, _ in links}
    links.extend((url, "") for value in re.findall(r"https?://[^\s<>\"']+", plain_text)
                 for url in [value.rstrip(".,;!?)>")] if url not in html_urls)
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
        try:
            query = parse_qsl(urlsplit(url).query, keep_blank_values=True, max_num_fields=256)
        except ValueError:
            query = []
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
    headers = {str(name).lower(): value if isinstance(value, (list, tuple)) else [value]
               for name, value in headers.items()}
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
        if model.get("version") != 2 or calibration.get("model_id") != model.get("model_id"):
            raise ValueError("модель и калибровка не совпадают, нужно повторное обучение")
        for name in ("mean", "scale", "weights"):
            if len(model[name]) != len(FEATURE_NAMES) or not all(math.isfinite(x) for x in model[name]):
                raise ValueError(f"некорректные параметры модели: {name}")
        if any(x <= 0 for x in model["scale"]):
            raise ValueError("масштаб признаков должен быть положительным")
        if not math.isfinite(model["bias"]):
            raise ValueError("некорректное смещение модели")
        if not all(math.isfinite(calibration[name]) for name in ("weight", "bias", "threshold")):
            raise ValueError("повреждена калибровка")
        if not 0 <= calibration["threshold"] <= 1 or calibration["weight"] < 0:
            raise ValueError("некорректный порог или вес калибровки")
        self.model = model
        self.calibration = calibration

    def transform(self, features: dict) -> list:
        return [(x - mean) / scale for x, mean, scale in
                zip(feature_vector(features), self.model["mean"], self.model["scale"])]

    def decision(self, features: dict) -> float:
        return self.model["bias"] + sum(w * x for w, x in zip(self.model["weights"], self.transform(features)))

    def probability(self, features: dict) -> float:
        return sigmoid(self.calibration["weight"] * self.decision(features) + self.calibration["bias"])

    def contributions(self, features: dict) -> list:
        return [w * x * self.calibration["weight"] for w, x in
                zip(self.model["weights"], self.transform(features))]

    def explain(self, features: dict) -> list:
        values = self.contributions(features)
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


def find_evidence(html, contributions):
    spans = []
    names = {name for name, value in zip(FEATURE_NAMES, contributions) if abs(value) >= 0.01}
    for match in re.finditer(r"<!--[\s\S]*?-->|<[a-zA-Z][^>]*>", html):
        tag = match.group()
        feature = None
        if tag.startswith("<!--") and "comments_count" in names:
            feature = "comments_count"
        elif re.match(r"<a\b", tag, re.I) and "links_count" in names:
            feature = "links_count"
        elif re.match(r"<img\b", tag, re.I) and "images_count" in names:
            feature = "images_count"
        elif "style=" in tag.lower().replace(" ", "") and "inline_style_ratio" in names:
            feature = "inline_style_ratio"
        if feature:
            spans.append(dict(zone="html", start=match.start(), end=match.end(), feature=feature,
                              reason="элемент участвует в расчёте признака; это не точная разметка ИИ-правки"))
        if len(spans) == 5:
            break
    return spans


def analyze_email_from_bytes(raw_bytes: bytes, include_score: bool = True, threshold=None,
                             model_path=MODEL_PATH, calibration_path=CALIBRATION_PATH) -> dict:
    parsed = EmailParser().parse_from_bytes(raw_bytes)
    html = parsed.text_html
    features = extract_all_features(html, parsed.headers, parsed.text_plain)
    result = dict(subject=parsed.subject, method="html_features_logistic", has_html=bool(html),
                  html_ai_index=None, html_ai_score=None, html_ai_verdict=None,
                  html_ai_status="not_evaluated" if html else "no_html", html_ai_evidence=[],
                  html_ai_reasons=["оценка отключена"] if html else ["нет html-части, признаки заголовков и ссылок сохранены"],
                  parse_warnings=parsed.warnings, **features)
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("порог должен быть от 0 до 1")
    if not html or not include_score:
        return result
    if not Path(model_path).is_file() or not Path(calibration_path).is_file():
        result.update(html_ai_status="not_trained", html_ai_reasons=["нужны обучение на парных HTML-письмах и калибровка"])
        return result
    try:
        classifier = HtmlClassifier(json.loads(Path(model_path).read_text(encoding="utf-8")),
                                    json.loads(Path(calibration_path).read_text(encoding="utf-8")))
        probability = classifier.probability(features)
        limit = classifier.calibration["threshold"] if threshold is None else threshold
        result.update(html_ai_index=round(probability, 6), html_ai_score=round(probability * 100, 1),
                      html_ai_verdict="AI-Edited" if probability >= limit else "Human-Written",
                      html_ai_threshold=limit, html_ai_status="ok", html_ai_reasons=classifier.explain(features),
                      html_ai_evidence=find_evidence(html, classifier.contributions(features)),
                      html_ai_model_id=classifier.model["model_id"])
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        result.update(html_ai_status="model_error", html_ai_reasons=[f"{type(error).__name__}: {error}"])
    return result


def analyze_email_from_file(file_path: str, include_score: bool = True, **options) -> dict:
    return analyze_email_from_bytes(Path(file_path).read_bytes(), include_score, **options)


def train_classifier(dataset_dir=DATASET_PATH, model_path=MODEL_PATH, calibration_path=CALIBRATION_PATH,
                     report_path=REPORT_PATH, max_fpr=0.2) -> dict:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise ValueError("нет парного HTML-датасета: сначала создай его через scripts/build_html_dataset.py")
    print("чтение парных писем с html...", flush=True)
    records = read_dataset(dataset_dir)
    text_hashes = {}
    for row in records:
        values = analyze_email_from_file(str(row["path"]), include_score=False)
        if not values["has_html"]:
            raise ValueError(f"в парном датасете нет html: {row['filename']}")
        parsed = EmailParser().parse_from_file(str(row["path"]))
        digest = hashlib.sha256(re.sub(r"\s+", " ", parsed.text_html).encode("utf-8")).hexdigest()
        if digest in text_hashes and text_hashes[digest] != row["split"]:
            raise ValueError("одинаковый html попал в разные разбиения")
        text_hashes[digest] = row["split"]
        row["features"] = values
    parts = {split: [row for row in records if row["split"] == split] for split in ("train", "validation", "test")}
    train = parts["train"]
    vectors = [feature_vector(row["features"]) for row in train]
    sample_weights = [row["weight"] for row in train]
    total = sum(sample_weights)
    mean = [sum(v[i] * w for v, w in zip(vectors, sample_weights)) / total for i in range(len(FEATURE_NAMES))]
    scale = [max(math.sqrt(sum(w * (v[i] - mean[i]) ** 2 for v, w in zip(vectors, sample_weights)) / total), 1e-8)
             for i in range(len(FEATURE_NAMES))]
    scaled = [[(x - m) / s for x, m, s in zip(vector, mean, scale)] for vector in vectors]
    print(f"обучение на {len(train)} письмах...", flush=True)
    weights, bias = fit_logistic(scaled, [row["target"] for row in train], sample_weights)
    model = dict(version=2, feature_names=list(FEATURE_NAMES), weights=weights, bias=bias, mean=mean, scale=scale,
                 seed=42, dataset=str(dataset_dir.resolve()), splits=split_summary(records),
                 train_groups=sorted({row["group_id"] for row in train}),
                 scope="парные оригинальные и отредактированные HTML-письма; это не spam/ham-классификатор")
    model["model_id"] = hashlib.sha256(json.dumps(model, sort_keys=True).encode("utf-8")).hexdigest()
    classifier = HtmlClassifier(model, dict(model_id=model["model_id"], weight=1.0, bias=0.0, threshold=0.5))
    validation = parts["validation"]
    calibration = calibrate([classifier.decision(row["features"]) for row in validation],
                            [row["target"] for row in validation], [row["weight"] for row in validation], max_fpr)
    calibration["model_id"] = model["model_id"]
    classifier.calibration = calibration
    test = parts["test"]
    metrics = calculate_metrics([classifier.probability(row["features"]) for row in test],
                                [row["target"] for row in test], [row["weight"] for row in test], calibration["threshold"])
    report = dict(model_id=model["model_id"], dataset=str(dataset_dir.resolve()), splits=model["splits"],
                  validation=calibration["validation_metrics"], test=metrics, threshold=calibration["threshold"],
                  meets_quality_target=metrics["f1"] >= 0.7 and metrics["roc_auc"] >= 0.7 and metrics["fpr"] <= 0.2,
                  note="тест не участвовал в обучении, калибровке и выборе порога; интервалы объясняют признаки, а не заменяют локализацию ИИ")
    save_json(model_path, model)
    save_json(calibration_path, calibration)
    save_json(report_path, report)
    print(f"модель: {model_path}\nкалибровка: {calibration_path}\nотчёт: {report_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("file", nargs="?", type=Path, help="письмо или папка с eml")
    cli.add_argument("--train", action="store_true")
    cli.add_argument("--dataset", type=Path, default=DATASET_PATH)
    cli.add_argument("--output", type=Path)
    cli.add_argument("--model", type=Path, default=MODEL_PATH)
    cli.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    cli.add_argument("--report", type=Path, default=REPORT_PATH)
    cli.add_argument("--threshold", type=float)
    cli.add_argument("--max-fpr", type=float, default=0.2)
    args = cli.parse_args()
    if args.train and args.file:
        cli.error("выбери обучение или анализ письма")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        cli.error("--threshold должен быть от 0 до 1")
    if not 0 <= args.max_fpr <= 1:
        cli.error("--max-fpr должен быть от 0 до 1")
    try:
        if args.train or args.file is None:
            result = train_classifier(args.dataset, args.model, args.calibration, args.report, args.max_fpr)
        else:
            result = analyze_path(args.file, analyze_email_from_file, "html_ai_status", threshold=args.threshold,
                                  model_path=args.model, calibration_path=args.calibration)
        print_json(result, args.output)
        return 0
    except (OSError, ValueError, KeyError, MessageError, RecursionError) as error:
        print_error(error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
