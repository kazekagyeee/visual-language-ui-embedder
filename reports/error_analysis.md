# Error analysis

## Query: покажи Входной контроль

- type: single
- recall: 0.0000
- mrr: 0.0000
- time_sec: 0.2383

### Targets

```json
[
  {
    "text": "Входной контроль",
    "normalized_text": "входной контроль",
    "target_pages": [
      1,
      3,
      5,
      7,
      24
    ],
    "ui_type": "sidebar_item"
  }
]
```

### Predictions

```json
[
  {
    "text": "АРМ Входной контроль; Создать:",
    "normalized_text": "арм входной контроль создать",
    "page": 14,
    "ui_type": "button",
    "score": 1.2233849465847015,
    "final_score": 3.9967698931694033,
    "siamese_score": 0.22338494658470154,
    "eval_score": 8.996769893169404
  }
]
```

## Query: где находится Дата

- type: single
- recall: 0.0000
- mrr: 0.0000
- time_sec: 0.2533

### Targets

```json
[
  {
    "text": "Дата",
    "normalized_text": "дата",
    "target_pages": [
      15,
      17,
      18,
      22,
      24,
      26,
      33
    ],
    "ui_type": "hyperlink"
  }
]
```

### Predictions

```json
[
  {
    "text": "Дата поставки:",
    "normalized_text": "дата поставки",
    "page": 25,
    "ui_type": "hyperlink",
    "score": 0.45932912826538086,
    "final_score": 2.468658256530762,
    "siamese_score": -0.5406708717346191,
    "eval_score": 7.468658256530762
  }
]
```

## Query: покажи Отчеты

- type: single
- recall: 0.0000
- mrr: 0.0000
- time_sec: 0.2368

### Targets

```json
[
  {
    "text": "Отчеты",
    "normalized_text": "отчеты",
    "target_pages": [
      9,
      11,
      33,
      35,
      36,
      37,
      38,
      39
    ],
    "ui_type": "button"
  }
]
```

### Predictions

```json
[
  {
    "text": "Отчеты по закупкам",
    "normalized_text": "отчеты по закупкам",
    "page": 12,
    "ui_type": "button",
    "score": 0.31911367177963257,
    "final_score": 2.188227343559265,
    "siamese_score": -0.6808863282203674,
    "eval_score": 7.188227343559265
  }
]
```

## Query: найди элемент ГОСТ

- type: single
- recall: 0.0000
- mrr: 0.0000
- time_sec: 0.6299

### Targets

```json
[
  {
    "text": "ГОСТ",
    "normalized_text": "гост",
    "target_pages": [
      8,
      15,
      16,
      17,
      19,
      20,
      26
    ],
    "ui_type": "hyperlink"
  }
]
```

### Predictions

```json
[
  {
    "text": "ГОСТы",
    "normalized_text": "гостыы",
    "page": 7,
    "ui_type": "hyperlink",
    "score": 1.4623022079467773,
    "final_score": 3.474604415893555,
    "siamese_score": 0.46230220794677734,
    "eval_score": 8.474604415893555
  },
  {
    "text": "ГОСТы",
    "normalized_text": "гостыы",
    "page": 7,
    "ui_type": "hyperlink",
    "score": 1.328535795211792,
    "final_score": 3.2070715904235843,
    "siamese_score": 0.328535795211792,
    "eval_score": 8.207071590423585
  },
  {
    "text": "ГОСТ заполнястся реквизитами:",
    "normalized_text": "госты заполнястся реквизитами",
    "page": 7,
    "ui_type": "hyperlink",
    "score": 0.6668617725372314,
    "final_score": 2.883723545074463,
    "siamese_score": -0.33313822746276855,
    "eval_score": 7.883723545074464
  }
]
```

## Query: покажи Заявки на контроль и Входной контроль

- type: multi
- recall: 0.5000
- mrr: 0.5000
- time_sec: 0.4700

### Targets

```json
[
  {
    "text": "Заявки на контроль",
    "normalized_text": "заявки на контроль",
    "target_pages": [
      1,
      3,
      5,
      7,
      14,
      24
    ],
    "ui_type": "hyperlink"
  },
  {
    "text": "Входной контроль",
    "normalized_text": "входной контроль",
    "target_pages": [
      1,
      3,
      5,
      7,
      24
    ],
    "ui_type": "sidebar_item"
  }
]
```

### Predictions

```json
[
  {
    "text": "Заявки на контроль",
    "normalized_text": "заявки на контроль",
    "page": 14,
    "ui_type": "hyperlink",
    "score": 1.5241360455751418,
    "final_score": 4.998272091150284,
    "siamese_score": 0.1241360455751419,
    "eval_score": 15.998272091150284
  },
  {
    "text": "АРМ Входной контроль; Создать:",
    "normalized_text": "арм входной контроль создать",
    "page": 14,
    "ui_type": "button",
    "score": 1.2233849465847015,
    "final_score": 3.9967698931694033,
    "siamese_score": 0.22338494658470154,
    "eval_score": 8.996769893169404
  }
]
```

