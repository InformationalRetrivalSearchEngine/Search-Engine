
from math import log, sqrt
from collections import defaultdict
import nltk
nltk.download('punkt')

inverted_index = defaultdict(list)
nos_of_documents = 1553
vects_for_docs = [] 
document_freq_vect = {}  



def iterate_over_all_docs():
    for i in range(nos_of_documents - 1):
        doc_text = get_document_text_from_doc_id(i)
        token_list = get_tokenized_and_normalized_list(doc_text)
        vect = create_vector(token_list)
        vects_for_docs.append(vect)


def create_vector_from_query(l1):
    vect = {}
    for token in l1:
        if token in vect:
            vect[token] += 1.0
        else:
            vect[token] = 1.0
    return vect



def generate_inverted_index():
    count1 = 0
    for vector in vects_for_docs:
        for word1 in vector:
            inverted_index[word1].append(count1)
        count1 += 1



def create_tf_idf_vector():
    vect_length = 0.0
    for vect in vects_for_docs:
        for word1 in vect:
            word_freq = vect[word1]
            temp = calc_tf_idf(word1, word_freq)
            vect[word1] = temp
            vect_length += temp ** 2

        vect_length = sqrt(vect_length)
        for word1 in vect:
            vect[word1] /= vect_length



def get_tf_idf_from_query_vect(query_vector1):
    vect_length = 0.0
    for word1 in query_vector1:
        word_freq = query_vector1[word1]
        if word1 in document_freq_vect:  
            query_vector1[word1] = calc_tf_idf(word1, word_freq)
        else:
            query_vector1[word1] = log(1 + word_freq) * log(
                nos_of_documents)  

        vect_length += query_vector1[word1] ** 2
    vect_length = sqrt(vect_length)
    if vect_length != 0:
        for word1 in query_vector1:
            query_vector1[word1] /= vect_length



def calc_tf_idf(word1, word_freq):
    return log(1 + word_freq) * log(nos_of_documents / document_freq_vect[word1])




def get_dot_product(vector1, vector2):
    if len(vector1) > len(vector2):  
        temp = vector1
        vector1 = vector2
        vector2 = temp
    keys1 = vector1.keys()
    keys2 = vector2.keys()
    sum = 0
    for i in keys1:
        if i in keys2:
            sum += vector1[i] * vector2[i]
    return sum


def get_tokenized_and_normalized_list(doc_text):
   


    tokens = nltk.word_tokenize(doc_text)
    ps = nltk.stem.PorterStemmer()
    stemmed = []
    for words in tokens:
        stemmed.append(ps.stem(words))
    return stemmed


def create_vector(l1):
    vect = {}  
    global document_freq_vect

    for token in l1:
        if token in vect:
            vect[token] += 1
        else:
            vect[token] = 1
            if token in document_freq_vect:
                document_freq_vect[token] += 1
            else:
                document_freq_vect[token] = 1
    return vect


def get_document_text_from_doc_id(doc_id):
   
    try:
        str1 = open("corpus/doc" + str(doc_id).zfill(4)).read()
    except:
        str1 = ""
    return str1

def get_result_from_query_vect(query_vector1):
    parsed_list = []
    for i in range(nos_of_documents - 1):
        dot_prod = get_dot_product(query_vector1, vects_for_docs[i])
        parsed_list.append((i, dot_prod))
        parsed_list = sorted(parsed_list, key=lambda x: x[1])
    return parsed_list


iterate_over_all_docs()


generate_inverted_index()


create_tf_idf_vector()

print("Here is a list of 15 tokens along with its docIds (sorted) in the inverted index")
count = 1
for word in inverted_index:
    if count >= 16:
        break
    print('token num ' + str(count) + ': ' + word + ': '),
    for docId in inverted_index[word]:
        print(str(docId) + ', '),
    print()
    count += 1

print()
while True:
    query = input("Please enter your query....")
    if len(query) == 0:
        break
    query_list = get_tokenized_and_normalized_list(query)
    query_vector = create_vector_from_query(query_list)
    get_tf_idf_from_query_vect(query_vector)
    result_set = get_result_from_query_vect(query_vector)

    for tup in result_set:
        print("The docid is " + str(tup[0]).zfill(4) + " and the weight is " + str(tup[1]))
    print()
