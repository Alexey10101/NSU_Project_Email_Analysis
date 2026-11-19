"""
собирает результаты анализаторов письма в один json.
нейросетевой анализ включается отдельно, так как требует весов моделей.
"""

import argparse
import json
import sys
from pathlib import Path

import boosting_analyze
import html_analyze
import image_analyze
import text_analyze
from model_utils import analyze_path, print_error, print_json


def analyze_email(file_path, image_model=None, boosting_model=None, neural=None):
    text = text_analyze.analyze_email_from_file(file_path)
    text.pop("cleaned_text", None)
    result = dict(text=text, html=html_analyze.analyze_email_from_file(file_path),
                  images=image_analyze.analyze_email_from_file(file_path, image_model),
                  catboost=boosting_analyze.analyze_email_from_file(file_path, classifier=boosting_model))
    if neural is not None:
        result["neural"] = neural(file_path)
        result["neural"].pop("cleaned_text", None)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("file", type=Path, help="письмо eml или папка")
    cli.add_argument("--output", type=Path)
    cli.add_argument("--neural", choices=("distilbert", "roberta"))
    cli.add_argument("--models", type=Path, default=boosting_analyze.MODEL_PATH.parent)
    args = cli.parse_args()
    try:
        neural = None
        if args.neural == "distilbert":
            from neural_analyze import analyze_email_from_file as neural
        elif args.neural == "roberta":
            from neural_roberta import analyze_email_from_file as neural
        path = image_analyze.MODEL_PATH
        images = image_analyze.ImageClassifier(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else None
        path = args.models / boosting_analyze.MODEL_PATH.name
        boosting = None
        if path.is_file() or args.models != boosting_analyze.MODEL_PATH.parent:
            boosting = boosting_analyze.BoostingClassifier(
                path, args.models / boosting_analyze.CALIBRATION_PATH.name,
                args.models / boosting_analyze.METADATA_PATH.name)
        result = analyze_path(args.file, analyze_email, image_model=images, boosting_model=boosting, neural=neural)
        print_json(result, args.output)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        print_error(error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
