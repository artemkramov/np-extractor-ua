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
See folder `examples` for more details. The description of the method that was used in the package can found in the work ([link](http://usim.org.ua/arch/2019/5/9.pdf)):

`Погорілий С.Д., Крамов А.А. Метод виявлення іменних груп в україномовних текстах. Control Systems and Computers. 2019. № 5. С. 48-59.`

Please cite this work if you use this package. Thank you! :)  

=====================================================
=====================================================

# Програмний пакет Python для екстракції іменних груп з україномовних текстів

Це простий пакет екстракції іменних груп з будь-яких україномовних текстів. Використовуйте інструмент `pip` для встановлення пакету.

## Встановлення
Використовуйте `pip` для встановлення:

`pip install noun-phrase-ua`

Увага: пакет містить декілька залежностей. Пакет `udpipe` потребує використання декількох додаткових утиліт для компіляції певних частин коду.

## Приклад використання
```
import noun_phrase_ua


nlp = noun_phrase_ua.NLP()
text = '"Послухати Зеленського, звичайно, цікаво з цієї точки зору. Тому я думаю, що дебати відбудуться. Але люди, в першу чергу будуть слухати Володимира Зеленського. Їх усіх, як я розумію, цікавить його особистість", - сказав Кучма, відповідаючи на питання журналістів.'
summary = nlp.extract_entities(text)

# summary["tokens"] contains list of tokens, summary["entities"] contains groups of indices
# than represent entities

```
Дивіться папку `examples` для подробиць використання. Опис методу, що використовується у пакеті, знаходиться в роботі ([посилання](http://usim.org.ua/arch/2019/5/9.pdf)):

`Погорілий С.Д., Крамов А.А. Метод виявлення іменних груп в україномовних текстах. Control Systems and Computers. 2019. № 5. С. 48-59.`

Будь ласка, цитуйте цю роботу, якщо використовуєте цей пакет. Дякуємо! :)

