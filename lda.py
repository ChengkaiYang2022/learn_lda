doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health."

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

from nltk.tokenize import RegexpTokenizer
import nltk.tokenize as tokenize
tokenizer = RegexpTokenizer('\w+')

raw = doc_c.lower()
tokens = tokenizer.tokenize(raw)

print(tokens)

from stop_words import get_stop_words

en_stop = get_stop_words('en')
stopped_tokens = [i for i in tokens if not i in en_stop]
print(stopped_tokens)

from nltk.stem.porter import PorterStemmer

p_stemmer = PorterStemmer()
texts = [p_stemmer.stem(i) for i in stopped_tokens]

print(texts)
from gensim import corpora, models
dictionary = corpora.Dictionary([texts])
print(dictionary.token2id)

corpus = [text for text in texts]
# print(corpus)
# corpus = [dictionary.doc2bow(text) for text in texts]
corpus = [dictionary.doc2bow(texts)]
print(corpus[0])
import gensim
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)
print(ldamodel.print_topics(num_topics=3, num_words=3))