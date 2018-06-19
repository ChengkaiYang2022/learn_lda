from gensim.corpora import Dictionary
text = [['我', '想吃', '大龙虾', '和', '烤猪蹄']]
dictionary = Dictionary(text)
print((dictionary))
doc = dictionary.doc2bow(['我', '想吃', '大龙虾', '和', '我','你','烤猪蹄'])
print(doc)
doc = dictionary.doc2bow(['男人', '都', '爱吃', '烤猪蹄'])
print(doc)

