import re
import os.path as op
import funcy as fp
from glob import glob

import pandas as pd

from gensim import models
from gensim.corpora import Dictionary, MmCorpus

import nltk
from nltk.corpus import stopwords

from spacy.lang.es import Spanish

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

MAIN_FOLDER = "."
DATA_PATHS = op.join(MAIN_FOLDER, "data/*.txt")
MODELS_FOLDER = op.join(MAIN_FOLDER, "models")

TOPICS_NUMBER = 10

nlp = Spanish()
stopwords = stopwords.words('spanish')

def tokenize_line(line):
	line = re.sub(r"[\(\[].*?[\)\]]", "", line)
	tokens = [token.text.lower().removesuffix('\n') for token in nlp(line)]
	tokens = [t for t in tokens if not 'm-ddhh-' in t]
	tokens = [t for t in tokens if len(t) > 2]
	tokens = set(tokens) - set(stopwords)
	return tokens
    
def tokenize(lines, token_size_filter=2):
    tokens = fp.mapcat(tokenize_line, lines)
    return [t for t in tokens if len(t) > token_size_filter]    

def load_doc(filename):
    group, doc_id = op.split(filename)
    doc_id = doc_id.split('.docx.txt')[0]
    with open(filename, errors='ignore') as f:
        doc = f.readlines()
    return {'group': group,
            'doc': doc,
            'tokens': tokenize(doc),
            'id': doc_id}

def nltk_stopwords():
    return set(nltk.corpus.stopwords.words('spanish'))

def prep_corpus(docs, additional_stopwords=set(), no_below=5, no_above=0.5):
  print('Building dictionary...')
  dictionary = Dictionary(docs)
  stopwords = nltk_stopwords().union(additional_stopwords)
  stopword_ids = map(dictionary.token2id.get, stopwords)
  dictionary.filter_tokens(stopword_ids)
  dictionary.compactify()
  dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
  dictionary.compactify()

  print('Building corpus...')
  corpus = [dictionary.doc2bow(doc) for doc in docs]

  return dictionary, corpus

if __name__ == "__main__":

	# load files and tokenize it
	docs = pd.DataFrame(list(map(load_doc, glob(DATA_PATHS)))).set_index(['group','id'])

	# create corpus
	dictionary, corpus = prep_corpus(docs['tokens'])

	### Laten Dirichlet Allocation (LDA) ###

	# fit the LDA model
	lda = models.ldamodel.LdaModel(
		corpus=corpus,
		id2word=dictionary,
		num_topics=TOPICS_NUMBER,
		passes=10)

	# save dictionary and LDA model
	MmCorpus.serialize(op.join(MODELS_FOLDER, 'ddhh.mm'), corpus)
	dictionary.save(op.join(MODELS_FOLDER, 'ddhh.dict'))
	lda.save(op.join(MODELS_FOLDER, 'ddhh.model'))

	# visualize LDA model
	vis_data = gensimvis.prepare(lda, corpus, dictionary)
	pyLDAvis.display(vis_data)


	### Hierarchical Dirichlet process (HDP) ###
	# fit the HDP model
	hdp = models.hdpmodel.HdpModel(corpus, dictionary, T=50)
	
	# save HDP model
	hdp.save(op.join(MODELS_FOLDER, 'hdp_ddhh.model'))

	# visualize HDP model
	vis_data = gensimvis.prepare(hdp, corpus, dictionary)
	pyLDAvis.display(vis_data)