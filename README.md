# Python package to extract NP from the Ukrainian language

This is a simple package to extract noun phrases from a raw Ukrainian text. Use `pip` tool to install the package:

## Installation
Use `pip` tool to install

`pip install noun-phrase-ua`

Caution: package has several dependencies. Package `udpipe` requires some extra utilities to compile some parts of code.

## Usage
```
import noun_phrase_ua


nlp = noun_phrase_ua.NLP()
text = '"Послухати Зеленського, звичайно, цікаво з цієї точки зору. Тому я думаю, що дебати відбудуться. Але люди, в першу чергу будуть слухати Володимира Зеленського. Їх усіх, як я розумію, цікавить його особистість", - сказав Кучма, відповідаючи на питання журналістів.'
summary = nlp.extract_entities(text)

# summary["tokens"] contains list of tokens, summary["entities"] contains groups of indices
# than represent entities

```
See folder `examples` for more details.

