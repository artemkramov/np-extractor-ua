from noun_phrase_ua.nlp import NLP
from mitie import *  # MITIE package is used to perform NER extraction


# Print summary which contains set of tokens and entities
def print_summary(summary_data):
    for entity_group in summary_data["entities"]:
        words = []
        for token_position in entity_group:
            words.append(summary_data["tokens"][token_position]["word"])
        print(" ".join(words))


# Set text
text = '"Послухати Зеленського, звичайно, цікаво з цієї точки зору. Тому я думаю, що дебати відбудуться. Але люди, в першу чергу будуть слухати Володимира Зеленського. Їх усіх, як я розумію, цікавить його особистість", - сказав Кучма, відповідаючи на питання журналістів.'

# Init object and run extraction
# by default NER model is not used
nlp = NLP()

# It is also possible to set path to your one-column CSV gazetteer file and UdPipe model
# nlp = NLP(gazetteer_path="path/to/your/gazetteer.csv", udpipe_path="path/to/your/udpipe_model")

summary = nlp.extract_entities(text)
print_summary(summary)

print(18 * "=")

# Load NER model
# NER model can be retrieved here: http://lang.org.ua/en/models/
ner = named_entity_extractor('uk_model.dat')

# Extract entities with NER
summary = nlp.extract_entities(text, ner)
print_summary(summary)
