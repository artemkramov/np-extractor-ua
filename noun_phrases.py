import nlp
import db
from word import Word
from db_text import DBText
from mitie import *


class NounPhrases:
    # Named-entity recognition model
    ner = None

    # Document from DB
    documents = {}

    # Database session marker
    db_session = db.session()

    # Init
    def __init__(self):
        # Load NER model
        print("Loading ner model...")
        self.ner = named_entity_extractor('uk_model.dat')

    # Get all documents to test
    def get_all_documents(self):

        # Get tokens from DB
        tokens = self.db_session.query(Word).all()

        documents = {}

        # Group every token by document
        for token in tokens:
            if not (token.DocumentID in documents):
                # if len(documents.keys()) > 4:
                #     break
                documents[token.DocumentID] = []
            documents[token.DocumentID].append(token)

        self.documents = documents

    @staticmethod
    def evaluate_precision(true_positives, false_positives):
        return true_positives / (true_positives + false_positives)

    @staticmethod
    def evaluate_recall(true_positives, false_negatives):
        return true_positives / (true_positives + false_negatives)

    @staticmethod
    def evaluate_f1(precision, recall):
        return 2 * precision * recall / (precision + recall)

    def print_results(self, true_positives, false_positives, false_negatives):
        precision = self.evaluate_precision(true_positives, false_positives)
        recall = self.evaluate_recall(true_positives, false_negatives)
        f1 = self.evaluate_f1(precision, recall)
        print("Precision: {0}".format(precision))
        print("Recall: {0}".format(recall))
        print("F1: {0}".format(f1))

    # Get metrics to estimate accuracy
    def get_metrics(self, without_ner=False, without_ud_parser=False):

        # Init all metrics
        true_positives_exact = 0
        false_positives_exact = 0
        false_negatives_exact = 0
        true_positive_partial = 0
        false_positives_partial = 0
        false_negatives_partial = 0

        # Loop through each document
        for document_id in self.documents:

            tokens_gold = self.documents[document_id]
            # print(document_id)

            # Get raw text of the document from DB
            document_db = self.db_session.query(DBText).filter(DBText.DocumentID == document_id).one()

            # Extract entities from the text given
            tagged_data = nlp.extract_entities(document_db.RawText, self.ner, False, without_ner, without_ud_parser)

            # Remove <root> token from the parsed result
            tokens_tag = [token for token in tagged_data['tokens'] if token['word'] != '<root>']
            if len(self.documents[document_id]) == len(tokens_tag):

                # Group golden tokens
                entities_gold = {}
                for idx, token in enumerate(tokens_gold):
                    if not (token.CoreferenceGroupID is None):
                        if not (token.CoreferenceGroupID in entities_gold):
                            entities_gold[token.CoreferenceGroupID] = []
                        entities_gold[token.CoreferenceGroupID].append({
                            'position': idx,
                            'token': token
                        })

                # Group tokens retrieved by entities
                entities = {}
                for idx, token in enumerate(tokens_tag):
                    if not (token['groupID'] is None):
                        if not (token['groupID'] in entities):
                            entities[token['groupID']] = []
                        entities[token['groupID']].append({
                            'position': idx,
                            'token': token
                        })

                # Init list to check if coreference group has been checked
                seen_exact = []
                seen_partial = []

                # Loop through each entity group to compare with gold
                for group_id in entities:

                    # Items of predicted entities
                    items = entities[group_id]

                    # Start and end positions of the entity
                    start = items[0]['position']
                    end = items[-1]['position'] + 1

                    # Get corresponding tokens from the gold set
                    tokens_selected_gold = tokens_gold[start:end]

                    # Init variables
                    is_gold_cluster = True
                    prev_token_gold = None
                    is_positive_partial = False

                    # Check if selected tokens form an entity
                    for idx, token_selected_gold in enumerate(tokens_selected_gold):

                        # If any token is not inside coreference group
                        # than stop search
                        if token_selected_gold.CoreferenceGroupID is None:
                            is_gold_cluster = False
                            break

                        # If token is single than break loop
                        if len(tokens_selected_gold) == 1:
                            break

                        # If previous token isn't empty and coreference groups aren't equal
                        # than stop search
                        if (not (
                                prev_token_gold is None)) and prev_token_gold.CoreferenceGroupID != token_selected_gold.CoreferenceGroupID:
                            is_gold_cluster = False
                            break
                        prev_token_gold = token_selected_gold

                    # If all tokens from gold set form group and its length equals to the length of the current entity
                    # than we can say that it's an exact match
                    if is_gold_cluster and len(entities_gold[tokens_selected_gold[0].CoreferenceGroupID]) == len(items):

                        # Increment counters
                        true_positives_exact += 1
                        seen_exact.append(tokens_selected_gold[0].CoreferenceGroupID)
                        is_positive_partial = True
                    else:
                        false_positives_exact += 1

                        # Check if the match is partial
                        # It means that the entity is located at the beginning or at the end of the gold cluster
                        if not (tokens_selected_gold[0].CoreferenceGroupID is None):
                            if start == 0 or tokens_selected_gold[0].CoreferenceGroupID != tokens_gold[start - 1]:
                                is_positive_partial = True
                        else:
                            if not (tokens_selected_gold[-1].CoreferenceGroupID is None):
                                if (end == len(tokens_gold) - 1) or tokens_selected_gold[-1].CoreferenceGroupID != tokens_gold[end + 1]:
                                    is_positive_partial = True

                    # Increment counters if the partial match is detected
                    if is_positive_partial:
                        true_positive_partial += 1
                        seen_partial.append(tokens_selected_gold[0].CoreferenceGroupID)
                    else:
                        false_positives_partial += 1

                # Count all coreference groups that were not detected
                for coreference_group_id in entities_gold:
                    if not (coreference_group_id in seen_exact):
                        false_negatives_exact += 1
                    if not (coreference_group_id in seen_partial):
                        false_negatives_partial += 1

        print("Exact data")
        self.print_results(true_positives_exact, false_positives_exact, false_negatives_exact)
        print("")
        print("Partial data")
        self.print_results(true_positive_partial, false_positives_partial, false_negatives_partial)


np = NounPhrases()
np.get_all_documents()

print("")
print("Metrics for all elements")
np.get_metrics()
print(25*"=")

print("")
print("Metrics without NER")
np.get_metrics(without_ner=True)
print(25*"=")

print("")
print("Metrics without UD parser")
np.get_metrics(without_ud_parser=True)
print(25*"=")
