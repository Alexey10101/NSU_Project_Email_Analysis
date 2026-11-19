"""
извлекает признаки изображений и оценивает их по модели genimage.
принимает отдельное изображение, письмо eml или папку с письмами.
"""

import argparse
import hashlib
import io
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from model_utils import analyze_path, calculate_metrics, calibrate, print_error, print_json, save_json
from parser import EmailParser, load_image_from_file
from text_analyze import fit_logistic, sigmoid


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "datasets/GenImage/data"
MODEL_PATH = BASE_DIR / "models/image_classifier.json"
REPORT_PATH = BASE_DIR / "training/reports/image_metrics.json"
FEATURE_LABELS = {
    "detail": "количество мелких деталей",
    "flat": "доля гладких участков",
    "contrast": "контраст яркости",
    "color": "разброс насыщенности",
    "edge_strength": "выраженность границ",
    "texture": "неравномерность мелких деталей",
}


def image_features(data):
    with Image.open(io.BytesIO(data)) as original:
        if original.width * original.height > 25_000_000:
            raise ValueError("изображение слишком большое для анализа")
        frame = ImageOps.exif_transpose(original).convert("RGB").resize((128, 128), Image.Resampling.LANCZOS)
    rgb = np.asarray(frame, dtype=np.float32) / 255
    luma = frame.convert("L")
    pixels = np.asarray(luma, dtype=np.float32) / 255
    smooth = np.asarray(luma.filter(ImageFilter.GaussianBlur(1)), dtype=np.float32) / 255
    detail = np.abs(pixels - smooth)
    blocks = detail.reshape(8, 16, 8, 16).mean(axis=(1, 3))
    small = np.asarray(luma.resize((9, 8), Image.Resampling.LANCZOS))
    perceptual_hash = int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), "big")
    features = dict(detail=float(detail.mean()), flat=float((detail < 2 / 255).mean()),
                    contrast=float(pixels.std()), color=float((rgb.max(axis=2) - rgb.min(axis=2)).std()),
                    edge_strength=float(np.abs(np.diff(pixels, axis=0)).mean() +
                                        np.abs(np.diff(pixels, axis=1)).mean()),
                    texture=float(blocks.std()))
    return features, perceptual_hash


class ImageClassifier:
    def __init__(self, model):
        if model["version"] != 1 or model["features"] != list(FEATURE_LABELS):
            raise ValueError("неподдерживаемый формат модели изображений")
        size = len(FEATURE_LABELS)
        if any(len(model[key]) != size or not all(math.isfinite(x) for x in model[key])
               for key in ("center", "scale", "weights")) or min(model["scale"]) <= 0:
            raise ValueError("повреждены признаки модели изображений")
        self.model = model

    def contributions(self, features):
        model = self.model
        return [(features[name] - center) / scale * weight for name, center, scale, weight
                in zip(FEATURE_LABELS, model["center"], model["scale"], model["weights"])]

    def decision(self, features):
        return self.model["bias"] + sum(self.contributions(features))

    def evaluate(self, features):
        cal = self.model["calibration"]
        probability = sigmoid(cal["weight"] * self.decision(features) + cal["bias"])
        values = dict(zip(FEATURE_LABELS, self.contributions(features)))
        reasons = []
        for name in sorted(values, key=lambda key: abs(values[key]), reverse=True)[:2]:
            contribution = values[name] * cal["weight"]
            if abs(contribution) > 0.001:
                direction = "повышает" if contribution > 0 else "снижает"
                reasons.append(f"{FEATURE_LABELS[name]} {direction} оценку")
        verdict = ("AI-Generated" if probability >= cal["threshold"] else "Natural") if cal.get("usable", True) else None
        return dict(image_ai_index=round(probability, 6), image_ai_score=round(probability * 100, 1),
                    image_ai_verdict=verdict, image_ai_status="ok" if verdict else "no_working_threshold",
                    image_ai_reasons=reasons, model_scope="Tiny-GenImage")


def analyze_image(image, classifier):
    result = dict(filename=image.filename, content_id=image.content_id, content_type=image.content_type,
                  image_ai_index=None, image_ai_score=None, image_ai_verdict=None, image_ai_reasons=[])
    try:
        features, _ = image_features(image.data)
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        return dict(result, image_ai_status="image_error", error=str(error))
    result.update(image_features=features, image_ai_status="not_trained")
    if classifier is not None:
        result.update(classifier.evaluate(features))
    return result


def analyze_email_from_file(file_path, classifier):
    parsed = EmailParser().parse_from_file(file_path)
    return dict(subject=parsed.subject, images=[analyze_image(image, classifier) for image in parsed.images],
                image_ai_status="ok" if parsed.images else "no_images", parse_warnings=parsed.warnings, external_images_count=len(parsed.image_urls))


def read_dataset(directory):
    directory = Path(directory).resolve()
    annotations = directory / "annotations.jsonl"
    strata = defaultdict(list)
    seen = []
    lines = [json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines() if line.strip()]
    part = {"train": 0, "validation": 1, "test": 2}
    for row in sorted(lines, key=lambda item: part[item["split"]]):
        path = (directory / row["file"]).resolve()
        if not path.is_relative_to(directory) or row["ai_label"] not in (0, 1):
            raise ValueError("неверный путь или метка изображения")
        data = path.read_bytes()
        if row.get("sha256") and hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"изображение не совпадает с разметкой: {path.name}")
        features, fingerprint = image_features(data)
        if any((old ^ fingerprint).bit_count() <= 4 for old in seen):
            continue
        seen.append(fingerprint)
        strata[row["split"], row["ai_label"], row.get("generator") or ""].append(features)

    splits = defaultdict(list)
    rng = random.Random(42)
    separate_test = any(split == "test" for split, _, _ in strata)
    for (split, label, _), rows in sorted(strata.items()):
        rng.shuffle(rows)
        if split == "validation" and not separate_test:
            cutoff = len(rows) // 2
            splits["test"].extend((row, label) for row in rows[cutoff:])
            rows = rows[:cutoff]
        splits[split].extend((row, label) for row in rows)
    if any({label for _, label in splits[split]} != {0, 1} for split in part):
        raise ValueError("в каждом разбиении нужны изображения обоих классов")
    return splits, len(lines) - len(seen), hashlib.sha256(annotations.read_bytes()).hexdigest()


def train_classifier(dataset, model_path, report_path):
    splits, removed, digest = read_dataset(dataset)
    train = splits["train"]
    vectors = np.array([[features[name] for name in FEATURE_LABELS] for features, _ in train])
    center = vectors.mean(axis=0)
    scale = np.maximum(vectors.std(axis=0), 1e-8)
    labels = [label for _, label in train]
    weights, bias = fit_logistic(((vectors - center) / scale).tolist(), labels, [1.0] * len(labels))
    model = dict(version=1, features=list(FEATURE_LABELS), center=center.tolist(),
                 scale=scale.tolist(), weights=weights, bias=bias)
    classifier = ImageClassifier(model)
    val_features, val_labels = zip(*splits["validation"])
    cal = calibrate([classifier.decision(row) for row in val_features], list(val_labels), [1.0] * len(val_labels))
    model["calibration"] = cal
    test_features, test_labels = zip(*splits["test"])
    probabilities = [sigmoid(cal["weight"] * classifier.decision(row) + cal["bias"]) for row in test_features]
    report = dict(dataset_annotations_sha256=digest, duplicates_removed=removed,
                  counts={split: len(rows) for split, rows in splits.items()}, threshold=cal["threshold"],
                  validation=cal["validation_metrics"],
                  test=calculate_metrics(probabilities, list(test_labels), threshold=cal["threshold"]),
                  scope="выборка GenImage; качество на изображениях из писем не проверялось")
    save_json(model_path, model)
    save_json(report_path, report)
    return report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("file", type=Path, nargs="?")
    cli.add_argument("--train", action="store_true")
    cli.add_argument("--dataset", type=Path, default=DATASET_PATH)
    cli.add_argument("--model", type=Path, default=MODEL_PATH)
    cli.add_argument("--report", type=Path, default=REPORT_PATH)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.train == (args.file is not None):
        cli.error("выбери --train или укажи файл или папку")
    try:
        if args.train:
            result = train_classifier(args.dataset, args.model, args.report)
        else:
            model = ImageClassifier(json.loads(args.model.read_text(encoding="utf-8"))) if args.model.is_file() else None
            if args.file.is_dir() or args.file.suffix.lower() == ".eml":
                result = analyze_path(args.file, analyze_email_from_file, "image_ai_status", classifier=model)
            else:
                result = analyze_image(load_image_from_file(args.file), model)
        print_json(result, args.output)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print_error(error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
