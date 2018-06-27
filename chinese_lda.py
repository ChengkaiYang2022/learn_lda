from gensim.corpora import Dictionary
import jieba
import gensim


def get_stop_words(file_name = "stop_words.txt"):
    f = open(file_name, "r", encoding="utf-8")
    result = list()
    for line in f.readlines():
        line = line.strip()
        result.append(line)
    return result


# 切词
def lda(line):
    tokens = list(jieba.cut(line))

    stop_word = get_stop_words()
    stopped_tokens = [i for i in tokens if i not in stop_word]
    dictionary = Dictionary([stopped_tokens])

    # 做语料库
    doc = [dictionary.doc2bow(tokens)]

    # 做统计
    ldamodel = gensim.models.ldamodel.LdaModel(doc, num_topics=2, id2word=dictionary, passes=20)
    result = list()
    for i in (ldamodel.print_topics(num_topics=2, num_words=10)):
        result.append(str(i[1]))
    return str(result)


