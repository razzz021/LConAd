# This implementation is based on DrQA https://github.com/facebookresearch/DrQA
# Ranker class suppose you read the whole corpus into memory
# remove caching reading and multiprocessing mechanism
# the doc-term matrix is optimized by scipy.sparse.scr_matrix

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import copy
import spacy
import numpy as np
import joblib
from scipy.sparse import csr_matrix
from typing import List
import os
import jieba


# def get_stop_words(path):
#     stop_words = []
#     with open(path, 'r', encoding='utf-8') as f:
#         for line in f.readlines():
#             stop_words.append(line.strip())
#     return set(stop_words)


# DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# STOP_WORDS = get_stop_words(os.path.join(DIR_PATH, "stopwords.txt"))
STOP_WORDS=set()


# TODO:add stop words filter
def filter_zh_fn(token_list: List):
    global STOP_WORDS
    stop_words = STOP_WORDS
    if set(token_list) < set(stop_words):
        return True
    else:
        return False

# TODO:add some normalizer
def normalize_text(doc):
    return doc


def build_analyzer(tokenizer_type, ngrams=1, filter_fn=filter_zh_fn, normalize_fn=normalize_text):
    tokenizer = Tokenizer(tokenizer_type=tokenizer_type, n=ngrams, filter_fn=filter_fn)

    def analyzer(doc):
        nonlocal tokenizer
        nonlocal normalize_fn
        doc = normalize_fn(doc)
        return tokenizer.ngrams(doc)
    return analyzer


class Tokenizer:
    def __init__(self, tokenizer_type="jieba", n=1, filter_fn=None):
        self.n = n
        self.tokenizer_type = tokenizer_type
        self.filter_fn = filter_fn
        
        if tokenizer_type == "jieba":
            self.tokenizer = jieba.cut
        elif tokenizer_type == "spacy":
            self.tokenizer = spacy.load("zh_core_web_sm")
        elif tokenizer_type == "basic":
            self.tokenizer = None
            
    def tokenize(self, text):
        if self.tokenizer_type == "jieba":
            return list(self.tokenizer(text))
        elif self.tokenizer_type == "spacy":
            return [token.text for token in self.tokenizer(text)]
        elif self.tokenizer_type == "basic":
            return [token for token in text.strip() if token]

    def ngrams(self, text):
        clean_text = text.replace('\n', ' ')
        tokens = self.tokenize(clean_text)
        res = []
        n = self.n
        for i in range(1, n + 1):
            res.extend(self._ngrams(tokens, i))
        return res

    def _ngrams(self, tokens, n):
        filter_fn = self.filter_fn
        if self.filter_fn is None:
            def filter_fn(x): return False
        res = []
        for i in range(0, len(tokens) - n + 1):
            if filter_fn(tokens[i:i + n]):
                continue
            res.append(" ".join(tokens[i:i + n]))
        return res

class BaseDocRanker(object):
    def __init__(self):
        self.doc_term_mat = None
        self.vectorizer = None
        self.transformer = None

    def save_tfidf_joblib(self, save_path):
        if self.doc_term_mat is None:
            raise ValueError('fit bofore save model')
        state = {"vectorizer": self.vectorizer,
                 'transformer': self.transformer,
                 'doc_term_mat': self.doc_term_mat,}
        save_name = os.path.join(save_path, "ranker.rk")
        joblib.dump(state, save_name)

    def load_tfidf_joblib(self, filename):
        state = joblib.load(filename)
        self.doc_term_mat = state['doc_term_mat']
        self.vectorizer = state['vectorizer']
        self.transformer = state['transformer']

class TfidfDocRanker(BaseDocRanker):
    def __init__(self, tokenizer_type="jieba", ngrams=2):
        analyzer = build_analyzer(tokenizer_type, ngrams=ngrams)
        self.vectorizer = CountVectorizer(analyzer=analyzer)
        self.transformer = TfidfTransformer()
        self.doc_term_mat = None

    def fit(self, corpus):
        self.doc_term_mat = self.vectorizer.fit_transform(corpus)
        self.transformer.fit(self.doc_term_mat)
        self.doc_term_mat = self.transformer.transform(self.doc_term_mat)

    def query_one(self, q: str, k=1) -> np.array:
        if self.doc_term_mat is None:
            raise ValueError('use fit before query or load weight before')
        if not isinstance(q, str):
            raise ValueError('query_one only accept str type')

        query = self._vectorize(q)
        res = self.doc_term_mat.dot(query.transpose())
        res = res.toarray().flatten()

        if len(res) <= k:
            o_sort = np.argsort(-res)
        else:
            o = np.argpartition(-res, k)[0:k]
            o_sort = o[np.argsort(-res[o])]

        doc_scores = res[o_sort]
        doc_ids = o_sort

        # when all documents score is 0
        # return -1 for index
        # return np.nan for scores
        if len(o_sort) == 0:
            doc_ids = [-1] * k
            doc_scores = [np.nan] * k

        if len(doc_ids) < k:
            tail = k - len(doc_ids)
            doc_ids = np.concatenate([doc_ids, -1 * np.ones(tail)])
            doc_scores = np.concatenate([doc_scores, np.zeros(tail)])

        # make sure return np.array
        doc_ids = np.array(doc_ids, dtype=np.longlong)
        doc_scores = np.array(doc_scores, dtype=np.single)
        return doc_ids, doc_scores

    def query_batch(self, batch_query, k=1):
        doc_ids = []
        doc_scores = []
        for query in batch_query:
            doc_id, doc_score = self.query_one(query, k)
            doc_ids.append(doc_id)
            doc_scores.append(doc_score)
        doc_ids = np.array(doc_ids, dtype=np.longlong)
        doc_scores = np.array(doc_scores, dtype=np.single)
        return doc_ids, doc_scores

    def _vectorize(self, q):
        return self.transformer.transform(self.vectorizer.transform([q]))

class Bm25DocRanker(TfidfDocRanker):
    def __init__(self, tokenizer_type="jieba", ngrams=2, b=0.75, k1=1.6):
        super().__init__(tokenizer_type, ngrams)
        self.b = b
        self.k1 = k1
        self.avgdl = None
        self.doc_len = None
        self.doclen_by_avgdl = None

    def fit(self, corpus):
        self.doc_term_mat = self.vectorizer.fit_transform(corpus)
        self.transformer.fit(self.doc_term_mat)
        doc_len = self.doc_term_mat.sum(1)
        self.doclen_by_avgdl = doc_len / doc_len.mean()

    def _bm25_score(self, q: csr_matrix):
        idf = self.transformer.idf_
        k1 = self.k1
        b = self.b
        doclen_by_avgdl = self.doclen_by_avgdl

        # calculating bm25 scores
        # naive implementation from https://en.wikipedia.org/wiki/Okapi_BM25
        # use query sparse rep to slice idf and doc-term-matrix
        idf = idf[q.indices]
        tf = self.doc_term_mat[:, q.indices]
        denom = tf + k1 * (1 - b + b * doclen_by_avgdl)
        scores = tf.multiply(np.broadcast_to(idf, tf.shape)) * (k1 + 1) / denom
        scores = scores.sum(1)
        return scores.A1

    def query_one(self, q: str, k=1):
        if self.doc_term_mat is None:
            raise ValueError('use fit before query or load weight before')
        if not isinstance(q, str):
            raise ValueError('query_one only accept str type')

        query = self._vectorize(q)
        res = self._bm25_score(query)
        if len(res) <= k:
            o_sort = np.argsort(-res)
        else:
            o = np.argpartition(-res, k)[0:k]
            o_sort = o[np.argsort(-res[o])]

        doc_scores = res[o_sort]
        doc_ids = o_sort

        # when all scores is 0
        # return -1 for index
        # return np.nan for scores
        if len(o_sort) == 0:
            doc_ids = [-1] * k
            doc_scores = [np.nan] * k

        if len(doc_ids) < k:
            tail = k - len(doc_ids)
            doc_ids = np.concatenate([doc_ids, -1 * np.ones(tail)])
            doc_scores = np.concatenate([doc_scores, np.zeros(tail)])

        doc_ids = np.array(doc_ids, dtype=np.longlong)
        doc_scores = np.array(doc_scores, dtype=np.single)
        return doc_ids, doc_scores

    def _vectorize(self, q):
        return self.vectorizer.transform([q])

class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.

        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_strings: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while idx < len(entities) and entities[idx] == ner_tag:
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups

class SpacyTokenizer(object):

    def __init__(self, language='zh', **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: spaCy model to use (either path, or keyword like 'en').
        """

        if language == "zh":
            model = kwargs.get('model', 'zh_core_web_sm')
        elif language == "en":
            model = kwargs.get('model', 'en_core_web_sm')
        else:
            raise ValueError(f'{language} not support, choose from zh or en')
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        nlp_kwargs = {"disable": ["parser"]}
        if not any([p in self.annotators for p in ['lemma', 'pos', 'ner']]):
            nlp_kwargs['disable'].append('tagger')
        if 'ner' not in self.annotators:
            nlp_kwargs['disable'].append('entity')
        self.nlp = spacy.load(model, **nlp_kwargs)

    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ')
        tokens = self.nlp(clean_text)
        data = []
        for i in range(len(tokens)):
            # Get whitespace
            start_ws = tokens[i].idx
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1].idx
            else:
                end_ws = tokens[i].idx + len(tokens[i].text)

            data.append((
                tokens[i].text,
                text[start_ws: end_ws],
                (tokens[i].idx, tokens[i].idx + len(tokens[i].text)),
                tokens[i].tag_,
                tokens[i].lemma_,
                tokens[i].ent_type_,
            ))

        # Set special option for non-entity tag: '' vs 'O' in spaCy
        return Tokens(data, self.annotators, opts={'non_ent': ''})

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()
