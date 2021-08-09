# -*- encoding: utf-8 -*-
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
from tqdm import tqdm
import common_utils
import logging
from sklearn.cluster import KMeans
import csv
from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt


TIME_LINE_DATA_FILE = './data/트윗크롤링(이상치 처리).csv'
SEARCH_RESULT_DATA_FILE = './data/계정크롤링(전처리-일반,타광고삭제) (계정정보,타임라인 트윗 모두 ' \
                          '포함).csv'
FREQUENT_WORDS_FILE = './data/word-frequency-result-100.csv'

RESULT_CLUSTERING_CSV_FILE = './data/clustering-result(모든_데이터_사용).csv'

ANALYSIS_DATA_RATIO = 1  # 분석에 사용될 데이터 비율 선택 (데이터 범위: 0~1)

okt = Okt()


def save_vocab_with_label(vocab_with_label):

    cluster_words = dict()

    for vl in vocab_with_label:
        if cluster_words.get(vl[1]) is None:
            cluster_words[vl[1]] = []
        else:
            cluster_words[vl[1]].append(vl[0])

    cluster_words = sorted(cluster_words.items())

    with open(RESULT_CLUSTERING_CSV_FILE, 'w') as file:
        writer = csv.writer(file)

        for key, value in cluster_words:
            key = '군집' + str(key+1)
            writer.writerow([key, value])


def get_words_indices(tf_idf_vectorizer, words):
    words_indices = []

    for word in words:
        words_index = get_word_index(tf_idf_vectorizer, word)

        if words_index is not None:
            words_indices.append(words_index)
        else:
            print("!! " + word)

    return words_indices


def get_word_index(tf_idf_vectorizer, word):
    index = int(tf_idf_vectorizer.vocabulary_.get(word))

    return index


def get_frequent_words():
    result = []

    df = pd.read_csv(FREQUENT_WORDS_FILE)

    for idx, row in df.iterrows():
        result.append(row['word'])

    return result


def get_corpus(_tweet_contents):
    corpus = []

    for tweet_content in tqdm(_tweet_contents):
        sentences_tag = okt.pos(tweet_content)

        noun_adj_list = []

        # tag가 명사이거나 형용사인 단어들만 noun_adj_list에 넣어준다.
        for word, tag in sentences_tag:
            if tag in ['Noun', 'Adjective'] and not common_utils.is_stop_words(word):
                noun_adj_list.append(word)

        sentence = ' '.join(noun_adj_list)

        # print(sentence)
        corpus.append(sentence)

    return corpus


if __name__ == '__main__':

    if 0 > ANALYSIS_DATA_RATIO or ANALYSIS_DATA_RATIO > 1:  # 잘못된 수치 값을 설정했을 때, 프로그램 종료
        logging.error(f'잘못된 ANALYSIS_DATA_RATIO 값: {ANALYSIS_DATA_RATIO}')
        exit(-1)

    tweet_data1 = pd.read_csv(TIME_LINE_DATA_FILE, encoding='cp949')
    tweet_data2 = pd.read_csv(SEARCH_RESULT_DATA_FILE, encoding='cp949')

    tweet_contents = tweet_data1[tweet_data1['y'] == 1]['tweet'].dropna()

    # 2개 데이터 합침
    tweet_contents = tweet_contents.append(tweet_data2['tweet'].dropna())

    num_row = int(tweet_contents.shape[0] * ANALYSIS_DATA_RATIO)
    print(f'분석에 사용할 데이터 개수: {num_row}')

    tweet_contents = tweet_contents[:num_row+1]

    corpus = get_corpus(tweet_contents)

    tf_idf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')

    transpose_tfidf_matrix = tf_idf_vectorizer.fit_transform(corpus).T

    frequent_words = get_frequent_words()
    frequent_words_indices = get_words_indices(tf_idf_vectorizer, frequent_words)
    print(f'빈출 단어 개수 : {len(frequent_words_indices)}')

    frequent_matrix = []
    for frequent_words_index in frequent_words_indices:
        frequent_matrix.append(transpose_tfidf_matrix.toarray()[frequent_words_index])

    model = TSNE(learning_rate=100)
    transformed = model.fit_transform(frequent_matrix)

    # 군집 계산
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(transformed)

    # 군집 시각화
    # xs = transformed[:, 0]
    # ys = transformed[:, 1]
    # plt.scatter(xs, ys, c=kmeans.labels_)
    # plt.show()

    vocab_with_label = list(zip(frequent_words, kmeans.labels_))

    save_vocab_with_label(vocab_with_label)
