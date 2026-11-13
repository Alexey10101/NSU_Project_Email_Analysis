"""
проверяет разбор ссылок, метрики, защиту от утечек и обучение HTML-модели.
тестовые пары созданы вручную только для проверки кода и не являются обучающим датасетом проекта.
"""

import json
import math
import sys
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch
from email.message import EmailMessage
from email.errors import MessageError
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "analysis"))
sys.path.insert(0, str(BASE_DIR / "scripts"))
sys.path.insert(0, str(BASE_DIR / "training"))

import html_analyze
from model_utils import calculate_metrics, calibrate, read_dataset
from build_html_dataset import check_local_url, replace_html, changed_spans
from build_html_dataset import prepare_dataset
from parser import EmailParser


def make_email(html, number=0):
    msg = EmailMessage()
    msg["Subject"] = f"проверочное письмо {number}"
    msg["From"] = "sender@example.org"
    msg["To"] = "recipient@example.org"
    msg.set_content("текст проверочного письма")
    msg.add_alternative(html, subtype="html")
    return msg.as_bytes()


class ModelTests(unittest.TestCase):
    def test_links_and_headers(self):
        html = '<a href="https://[broken">ошибка</a><a href="https://evil.example/?utm_source=x">good.example</a>'
        values = html_analyze.extract_all_features(html, {"From": "a@one.example", "Reply-To": ["b@two.example"],
                                                         "Received": ["a", "b"]}, "ещё https://plain.example/a")
        self.assertEqual(values["links_count"], 2)
        self.assertEqual(values["url_mismatch_count"], 1)
        self.assertEqual(values["tracking_links_count"], 1)
        self.assertEqual(values["received_count"], 2)
        self.assertEqual(values["reply_to_mismatch"], 1)
        self.assertTrue(all(math.isfinite(x) for x in values.values()))

    def test_missing_model_and_plain_email(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.json"
            values = html_analyze.analyze_email_from_bytes(make_email("<p>текст</p>"), model_path=missing, calibration_path=missing)
            self.assertEqual(values["html_ai_status"], "not_trained")
            self.assertIsNone(values["html_ai_score"])
            msg = EmailMessage()
            msg.set_content("только текст")
            self.assertEqual(html_analyze.analyze_email_from_bytes(msg.as_bytes())["html_ai_status"], "no_html")
            with self.assertRaises(ValueError):
                html_analyze.analyze_email_from_bytes(msg.as_bytes(), threshold=1.5)
            batch = Path(temporary) / "batch"
            batch.mkdir()
            (batch / "a.eml").write_bytes(b"broken")
            (batch / "b.eml").write_bytes(msg.as_bytes())
            output = Path(temporary) / "batch.json"
            values = [MessageError("повреждённый заголовок"), dict(html_ai_status="no_html")]
            with patch.object(sys, "argv", ["html_analyze.py", str(batch), "--output", str(output)]), \
                    patch("html_analyze.analyze_email_from_file", side_effect=values), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(html_analyze.main(), 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual([row["html_ai_status"] for row in result], ["parse_error", "no_html"])

    def test_weighted_metrics_and_threshold(self):
        metrics = calculate_metrics([0.5, 0.5], [0, 1], [2, 2])
        self.assertEqual(metrics["roc_auc"], 0.5)
        self.assertEqual(metrics["fpr"], 1)
        calibration = calibrate([-2, -1, 1, 2], [0, 0, 1, 1], [1] * 4, max_fpr=0)
        self.assertEqual(calibration["validation_metrics"]["fpr"], 0)
        self.assertEqual(calibration["fitted_on"], "validation")

    def test_local_generation_safety(self):
        for url in ("http://example.org/v1", "https://localhost.example.org/v1", "http://localhost:1234/v1?key=secret"):
            with self.assertRaises(ValueError):
                check_local_url(url)
        check_local_url("http://127.0.0.1:1234/v1")
        before = '<p>старый текст</p><a href="https://example.org">ссылка</a>'
        after = '<p style="color:blue">новый текст</p><a href="https://example.org">ссылка</a>'
        parsed = EmailParser().parse_from_bytes(replace_html(make_email(before), after))
        self.assertEqual(parsed.text_html, after)
        self.assertIn("новый текст", parsed.text_plain)
        self.assertNotIn("старый текст", parsed.text_plain)
        spans = changed_spans(before, parsed.text_html, "html")
        self.assertTrue(spans)
        self.assertTrue(all(0 <= span["start"] < span["end"] <= len(parsed.text_html) for span in spans))

    def test_training_and_group_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data"
            dataset.mkdir()
            annotations = []
            for index in range(30):
                split = "train" if index < 20 else "validation" if index < 25 else "test"
                for label, category in ((0, "human"), (1, "edited")):
                    html = f'<p>{index} проверочное письмо</p>' if not label else f'<div style="padding:10px"><p>{index} другая версия</p></div>'
                    name = f"{split}/{category}/sample_{index:06d}.eml"
                    path = dataset / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(make_email(html, index))
                    annotations.append(dict(file=name, split=split, group_id=f"group{index}", ai_label=label,
                                            generator="test_fixture" if label else None))
            annotation_path = dataset / "annotations.jsonl"
            annotation_path.write_text("\n".join(json.dumps(row) for row in annotations), encoding="utf-8")
            records = read_dataset(dataset)
            self.assertEqual(len(records), 60)
            model, calibration, report = root / "model.json", root / "calibration.json", root / "report.json"
            result = html_analyze.train_classifier(dataset, model, calibration, report)
            self.assertTrue(report.is_file())
            self.assertEqual(result["splits"]["train"]["groups"], 20)
            prediction = html_analyze.analyze_email_from_bytes(make_email('<p>новое письмо</p>'), model_path=model, calibration_path=calibration)
            self.assertEqual(prediction["html_ai_status"], "ok")
            self.assertTrue(0 <= prediction["html_ai_index"] <= 1)
            broken = json.loads(calibration.read_text(encoding="utf-8"))
            broken["model_id"] = "wrong"
            calibration.write_text(json.dumps(broken), encoding="utf-8")
            prediction = html_analyze.analyze_email_from_bytes(make_email('<p>новое письмо</p>'), model_path=model, calibration_path=calibration)
            self.assertEqual(prediction["html_ai_status"], "model_error")
            # отдельное обучение catboost проверяем только во временной папке
            from build_table import build_meta_dataset
            from train_boosting import train_model
            from boosting_analyze import BoostingClassifier, email_features
            table = root / "table.csv"
            build_meta_dataset(dataset_dir=dataset, output_path=table)
            trained = train_model(table, root / "models", root / "boosting_report.json", root / "logs", iterations=10)
            self.assertEqual(trained["splits"]["test"]["groups"], 5)
            classifier = BoostingClassifier(root / "models" / "catboost_email_model.json",
                                            root / "models" / "catboost_calibration.json",
                                            root / "models" / "catboost_metadata.json")
            _, features = email_features(make_email('<p>проверка загрузки модели</p>'))
            self.assertTrue(0 <= classifier.evaluate(features)["ai_index"] <= 1)
            with self.assertRaises(ValueError):
                classifier.evaluate(features, threshold=-1)
            # одинаковый group_id в train и test должен остановить обучение
            annotations[-1]["group_id"] = "group0"
            annotation_path.write_text("\n".join(json.dumps(row) for row in annotations), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "разные разбиения"):
                read_dataset(dataset)

    def test_local_dataset_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            for index in range(24):
                word = chr(97 + index)
                text = " ".join([word * 5] + ["слово"] * 24)
                (source / f"original_{index}.eml").write_bytes(make_email(f"<p>{text}</p>", index))

            def local_reply(url, payload=None, **options):
                if payload is None:
                    return dict(data=[dict(id="test_fixture")])
                html = payload["messages"][0]["content"].split("\nHTML:\n", 1)[1]
                edited = '<div style="padding:10px">' + html + '</div>'
                return dict(choices=[dict(message=dict(content=json.dumps(dict(html=edited))))])

            with patch("build_html_dataset.request_json", side_effect=local_reply):
                prepare_dataset(source, root / "output", limit=24)
            rows = read_dataset(root / "output")
            self.assertEqual(len(rows), 48)
            self.assertTrue(all(row["annotation"]["generated_spans"] for row in rows if row["target"]))


if __name__ == "__main__":
    unittest.main()
