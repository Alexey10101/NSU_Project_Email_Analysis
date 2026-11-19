# Анализ электронных писем

Программа извлекает признаки ИИ-правок из писем в формате EML.

- `analysis` — парсер и анализаторы текста, HTML и признаков письма.
- `training` — подготовка таблицы, обучение CatBoost и отчёты.
- `scripts` — загрузка датасетов и подготовка пар HTML-писем.
- `tests` — проверка кода, сравнение пары писем и оценка нейросетевого блока.
- `models` — сохранённые модели и калибровки.
- `datasets` — локальные датасеты и сведения об источниках.
- `info` — план и материалы проекта.

## Анализ письма и изображений

~~~text
python -m pip install -r requirements.txt
python analysis/email_analyze.py письмо.eml --output результат.json
python analysis/image_analyze.py изображение.jpg
python analysis/image_analyze.py --train
~~~

Общий отчёт содержит отдельные результаты текста, HTML, изображений и CatBoost. Анализатор изображений читает вложения и встроенные картинки из EML, включая изображения с идентификатором CID. Внешние картинки отмечаются отдельно. Модель изображений обучена на Tiny-GenImage. Метрики в [отчёте](training/reports/image_metrics.json) относятся к этой выборке. Для оценки на вложениях нужна отдельная размеченная выборка.

## Проверка нейросетевого анализа

Сохранился [вариант напарника](https://github.com/Alexey10101/NSU_Project_Email_Analysis/tree/bd79c68) с RoBERTa-large и Qwen2.5-1.5B. При наличии CUDA он использует GPU. Скрипты проверки и сборки таблицы ищут PASTED в `/content/datasets/PASTED/data` для Colab и в `datasets/PASTED/data` на ПК.

~~~text
python -m pip install -r training/requirements-roberta.txt
python tests/evaluate_neural.py --compare --output training/reports/neural_comparison.json
python tests/evaluate_roberta.py --plot training/reports/neural_roberta.png
python analysis/neural_analyze.py письмо.eml --roberta --calibration training/reports/neural_comparison.json
~~~

В проверке и при анализе по умолчанию используются 4 возмущения и seed 42. Проверка сравнивает полные пары, которые смогли оценить обе модели. Порог и калибровка подбираются на validation, итоговые метрики считаются на test. Без подходящего отчёта анализатор возвращает сырой балл. Если текст слишком короткий или RoBERTa не находит замен для английских слов, оценка остаётся пустой.

## CatBoost с нейросетевыми признаками

~~~text
python training/build_neural_table.py --limit 100
python training/train_neural_boosting.py
python training/train_neural_boosting.py --without-neural
python analysis/boosting_analyze.py письмо.eml --models models/neural
~~~

Обе модели CatBoost обучаются на одной таблице. Одна использует нейросетевые признаки, другая служит для сравнения. Экспериментальные модели сохраняются в `models/neural` и `models/comparison`. Оценка изображений пока выводится отдельно от итоговой оценки CatBoost. Подготовка HTML-пар и обучение основного CatBoost описаны в [инструкции](training/README.md).

[Отчёт раннего эксперимента](training/reports/archive/neural_before_roberta.png) относится к предыдущей версии нейросетевого анализа.
