from gensim.corpora import Dictionary

def get_stop_words(file_name = "stop_words.txt"):
    f = open(file_name, "r",encoding="utf-8")
    result = list()
    for line in f.readlines():
        line = line.strip()
        result.append(line)
    return result


import jieba
line1 = " 你好，我前两天被xxx骗了，交了100元的手续费，第二天给我打电话让我卡里存5万元，我感觉是骗子，就告诉他不贷款了，但是她说我违约了，要交1000元违约金，合同自然解除，我要了两次账号那边就关机了，一直联系不上，我合同上贷款5万，那边能不能起诉我啊"

# 切词
tokens = list(jieba.cut(line1))
print(tokens)

stop_word = get_stop_words()
print("不得已" in stop_word)
stopped_tokens = [i for i in tokens if i not in stop_word]
dictionary = Dictionary([stopped_tokens])
print((dictionary))

# 做语料库
doc = [dictionary.doc2bow(list(jieba.cut(line1)))]
print(doc)

# 做统计
import gensim
ldamodel = gensim.models.ldamodel.LdaModel(doc, num_topics=3, id2word=dictionary, passes=20)

for i in (ldamodel.print_topics(num_topics=3, num_words=8)):
    print(i)