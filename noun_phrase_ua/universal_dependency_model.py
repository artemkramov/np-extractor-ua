import ufal.udpipe


class UniversalDependencyModel:
    # udpipe compiled model
    model = None

    np_simple_tags_pre = ['ADJ', 'NUM', 'DET', 'ADV', 'ADP']
    np_noun_tag = 'NOUN'
    np_prop_tag = 'PROPN'
    np_pron_tag = 'PRON'
    np_x_tag = 'X'
    np_head_tags = [np_noun_tag, np_prop_tag, np_x_tag]
    np_optional_tags = ['CCONJ']
    np_simple_tags = np_simple_tags_pre[:]

    np_allowed_tags = []

    np_relation_nmod = 'nmod'
    np_relation_case = 'case'
    np_relation_obl = 'obl'

    np_relation_child = ['nmod', 'compound', 'fixed', 'flat']

    np_strip_symbols = [',', ')', '(', '-', '"', ':', '»', '«', '–']
    np_forbidden_child_symbols = [',', ')', '(', ':']

    def __init__(self, path):
        # Load model by the given path
        self.model = ufal.udpipe.Model.load(path)
        self.np_simple_tags.extend(['DET', 'PRON', 'PUNCT', 'SYM'])
        self.np_allowed_tags = self.np_simple_tags[:]
        self.np_allowed_tags.extend(self.np_head_tags)
        if not self.model:
            raise Exception("Cannot load model by the given path: %s" % path)

    def parse(self, sentence):
        self.model.parse(sentence, self.model.DEFAULT)

    def tokenize(self, text):
        """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self._read(text, tokenizer)

    def read(self, text, in_format):
        """Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
        input_format = ufal.udpipe.InputFormat.newInputFormat(in_format)
        if not input_format:
            raise Exception("Cannot create input format '%s'" % in_format)
        return self._read(text, input_format)

    def _read(self, text, input_format):
        input_format.setText(text)
        error = ufal.udpipe.ProcessingError()
        sentences = []

        sentence = ufal.udpipe.Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = ufal.udpipe.Sentence()
        if error.occurred():
            raise Exception(error.message)

        return sentences

    def tag(self, sentence):
        """Tag the given ufal.udpipe.Sentence (inplace)."""
        self.model.tag(sentence, self.model.DEFAULT)

    def write(self, sentences, out_format):
        """Write given ufal.udpipe.Sentence-s in the required format (conllu|horizontal|vertical)."""

        output_format = ufal.udpipe.OutputFormat.newOutputFormat(out_format)
        output = ''
        for sentence in sentences:
            output += output_format.writeSentence(sentence)
        output += output_format.finishDocument()

        return output

    def extract_noun_phrases(self, sentences, named_entities):
        sentences_groups = []
        token_offset = 0
        named_entities_indexes = []
        levels = {}
        for r in named_entities:
            named_entities_indexes.extend(list(r))

        # Loop through the sentences
        for s in sentences:
            i = 0
            word_root_index = -1

            # Loop through words of the sentence and find out root word
            root_index = -1
            while i < len(s.words):
                word = s.words[i]

                # Check if the head links to the <root> element
                # and set index
                if word.head == 0:
                    word_root_index = i
                if word.id == 0:
                    root_index = i
                i += 1

            # Retrieve root word
            word_root = s.words[word_root_index]

            # Find groups with head nouns and corresponding tokens
            groups = {}
            self.np_recursive_extractor(word_root, s.words, groups, named_entities_indexes, token_offset, None)

            # Find levels for each word
            levels.update(self.np_write_levels(s.words[root_index], s.words, 0, token_offset))

            # Fix token sequence inside each NP group
            # It is necessary to remove all spaces inside the group
            np_indexes = list(groups.keys())
            np_indexes.sort()
            sentence_group = []
            for np_index in np_indexes:

                # Get all token list of NP and sort it in ascending order
                group = groups[np_index]
                group.sort()

                # Find index of head token
                np_group_idx = group.index(np_index)

                # Init corrected group
                group_aligned = [np_index]

                # Loop till the end of list from the head word
                # and check if all numbers is located near each other
                i = np_group_idx + 1
                while i < len(group):
                    if group[i] == group[i - 1] + 1:
                        group_aligned.append(group[i])
                    else:
                        break
                    i += 1

                # Loop till the start of list from the head word
                # and check if all numbers is located near each other
                i = np_group_idx - 1
                while i >= 0:
                    if group[i] + 1 == group[i + 1]:
                        group_aligned.append(group[i])
                    else:
                        break
                    i -= 1

                # Sort aligned group
                group_aligned.sort()

                # Remove comma at the start and end of sequence
                self.np_strip(group_aligned, 'form', self.np_strip_symbols, s.words)

                # Remove prepositions from the start/end of phrase
                self.np_strip(group_aligned, 'upostag', self.np_optional_tags, s.words)

                # Append group as range
                is_proper_name = False
                if s.words[np_index].upostag == 'PROPN':
                    is_proper_name = True
                sentence_group.append({
                    'head_word': np_index + token_offset,
                    'items': range(group_aligned[0] + token_offset, group_aligned[-1] + token_offset + 1),
                    'is_proper_name': is_proper_name
                })

            # Append all groups of the sentence to the general collection
            sentences_groups.extend(sentence_group)
            token_offset += len(s.words)
        return sentences_groups, levels

    # Parse tag string (like Animacy=Inan|Case=Loc|Gender=Masc|Number=Sing
    @staticmethod
    def parse_morphological_tag(tag_string):
        # Split by delimiter to separate each string
        morphology_strings = tag_string.split('|')
        morphology_attributes = []
        for morphology_string in morphology_strings:
            # Split each string to fetch attribute and its value
            morphology_attribute = morphology_string.split('=')
            morphology_attributes.append(morphology_attribute)
        return morphology_attributes

    # Fetch morphological feature by the given name
    def fetch_morphological_feature(self, tag_string, feature_name):
        morphology_attributes = self.parse_morphological_tag(tag_string)
        return [attribute_data[1] for attribute_data in morphology_attributes if attribute_data[0] == feature_name]

    # Parse tag string (like Animacy=Inan|Case=Loc|Gender=Masc|Number=Sing
    # https://universaldependencies.org/u/feat/index.html
    def parse_tag(self, tag_string):
        morphology_attributes = self.parse_morphological_tag(tag_string)

        # Set initial data
        is_plural = False
        gender = None

        for morphology in morphology_attributes:

            # Extract gender
            if morphology[0] == 'Gender':
                gender = morphology[1]

            # Extract plurality
            if morphology[0] == 'Number' and morphology[1] == 'Plur':
                is_plural = True

        return is_plural, gender

    # Strip noun phrase with given filter and parameter
    @staticmethod
    def np_strip(group, attribute, detect_group, words):
        if len(group) > 1 and getattr(words[group[0]], attribute) in detect_group:
            del group[0]
        if len(group) > 1 and getattr(words[group[-1]], attribute) in detect_group:
            del group[-1]

    # Traverse through tree and write levels of each word
    def np_write_levels(self, word, words, level, offset):
        levels = {
            word.id + offset: level
        }
        level += 1
        i = 0
        while i < len(word.children):
            levels.update(self.np_write_levels(words[word.children[i]], words, level, offset))
            i += 1
        return levels

    # Traverse the syntactic tree and find noun phrases
    def np_recursive_extractor(self, word, words, groups, named_entity_indexes, offset, head_id=None):
        if groups is None:
            groups = {}
        i = 0
        new_head_id = None
        token_id = word.id

        # Part of speech - NOUN, ADJ, VERB etc. - https://universaldependencies.org/u/pos/index.html
        u_pos_tag = word.upostag

        # Dependency relation to the head (amod, nmod) - https://universaldependencies.org/u/dep/index.html
        deprel = word.deprel

        is_object = False

        # Check if the parent id is passed and POS tag is allowed
        if (not (head_id is None)) and u_pos_tag in self.np_simple_tags:

            # Check if its children does'nt have the comma
            if len(word.children) > 0 and words[word.children[0]].form in self.np_forbidden_child_symbols:
                new_head_id = None
            else:
                new_head_id = head_id

            # Check if the token is located after head
            # and also if this tag with such POS is allowed before
            if token_id > head_id and token_id in words[head_id].children and u_pos_tag in self.np_simple_tags_pre:
                new_head_id = None

            if u_pos_tag == self.np_pron_tag and self.fetch_morphological_feature(word.feats, 'PronType') == ['Prs']:
                new_head_id = None

            # token_case = self.fetch_morphological_feature(word.feats, 'Case')
            # head_case = self.fetch_morphological_feature(words[head_id].feats, 'Case')
            # if token_case != head_case:
            #     new_head_id = None

        # Separately we analyze the noun and proper name
        if u_pos_tag in self.np_head_tags:
            new_head_id = token_id

            is_object = True
            # If it is necessary to add the current token to another group
            if not (head_id is None):
                # Check if we can add current noun to another NP
                # Firstly check if the type of the relation is in relation array
                if deprel.split(':')[0] in self.np_relation_child and words[head_id].deprel != self.np_relation_obl:

                    # Check if between these words all allowed POS tags
                    is_allowed_to_inherit = True

                    # Index of the next word after the head word
                    start = head_id + 1

                    # Imagine that noun can inherit just parent located before it
                    if start > token_id:
                        is_allowed_to_inherit = False
                    else:
                        # Loop through words and check it is possible to create that inheritance
                        while start < token_id:
                            if not (words[start].upostag in self.np_allowed_tags):
                                is_allowed_to_inherit = False
                                break
                            start += 1

                    # If current token is X element
                    # Than ignore Case element
                    if u_pos_tag != self.np_x_tag:
                        token_case = self.fetch_morphological_feature(word.feats, 'Case')
                        head_case = self.fetch_morphological_feature(words[head_id].feats, 'Case')

                        # Cannot connect dative Case if the parent has another
                        if token_case != head_case and token_case == ['Dat']:
                            is_allowed_to_inherit = False

                        # For nmod relation check if the Case morphological feature is different for objects
                        # Also allow inheritance of objects which are located near each other
                        if deprel == self.np_relation_nmod:
                            if token_case == 'Nom' and u_pos_tag == self.np_noun_tag and \
                                    words[head_id].upostag == self.np_noun_tag:
                                is_allowed_to_inherit = False

                            if words[head_id].deprel == self.np_relation_nmod:
                                i = head_id + 1
                                while i < token_id:
                                    if (not (words[i].upostag in self.np_head_tags)) or words[i].deprel != self.np_relation_nmod:
                                        is_allowed_to_inherit = False
                                        break
                                    i += 1

                    if is_allowed_to_inherit:
                        new_head_id = head_id

        # If the head word is set than add current token to its group
        if not (new_head_id is None):
            # Check if head_id and token_id aren't already inside named entities
            if (not ((token_id + offset) in named_entity_indexes)) and (
                    not ((new_head_id + offset) in named_entity_indexes)):
                self.np_push_to_group(groups, new_head_id, token_id)

        # Set default children parts as empty and all
        all_children = [[], word.children[:]]

        if (not (new_head_id is None)) and is_object:
            # Change the tree reverse order: from the center to left and right
            # Split children on left and right parts
            left_children = [x for x in word.children if x < new_head_id]
            right_children = [x for x in word.children if x > new_head_id]

            # Sort part of children
            # Left part is sorted in reverse order because we move from the middle to left
            left_children.sort(reverse=True)
            right_children.sort()

            # Concatenate children parts
            all_children = [left_children, right_children]

        # Loop through each part
        for children_part in all_children:

            # Set head_id as proposed new_head_id
            current_new_head_id = new_head_id

            # Loop through each children (word)
            for children_index in children_part:

                # Get the set head ID
                # If the head ID doesn't equal to proposed than we break its sequence of head ID
                children_head_id = self.np_recursive_extractor(words[children_index], words, groups,
                                                               named_entity_indexes, offset,
                                                               current_new_head_id)
                if children_head_id != new_head_id:
                    current_new_head_id = None

        return new_head_id

    @staticmethod
    def np_push_to_group(groups, head_id, token_id):
        # Create head group if it doesn't exist
        if not (head_id in groups):
            groups[head_id] = []
        groups[head_id].append(token_id)
