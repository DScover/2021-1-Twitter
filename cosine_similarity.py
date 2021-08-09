# -*- encoding: utf-8 -*-
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
from tqdm import tqdm
import common_utils
import logging
from sklearn.metrics.pairwise import cosine_similarity

TIME_LINE_DATA_FILE = './data/트윗크롤링(이상치 처리).csv'
FREQUENT_WORDS_FILE = './data/word-frequency-result-100.csv'

RESULT_COSINE_SIMILARITY_FILE = './data/cosine-similarity-result-100.csv'

ANALYSIS_DATA_RATIO = 0.01  # 분석에 사용될 데이터 비율 선택 (데이터 범위: 0~1)

okt = Okt()


def get_words_indices(tf_idf_vectorizer, words):
    words_indices = []

    for word in words:
        words_index = get_word_index(tf_idf_vectorizer, word)

        if words_index is not None:
            words_indices.append(words_index)

    return words_indices


def get_word_index(tf_idf_vectorizer, word):
    _index = tf_idf_vectorizer.vocabulary_.get(word)

    if _index is None:
        logging.error(f'cannot get {word} index')
        return None

    return int(_index)


def get_frequent_words():
    result = []

    df = pd.read_csv(FREQUENT_WORDS_FILE)

    for idx, row in df.iterrows():

        result.append(row['word'])

    return result


def get_corpus(_tweet_contents):
    corpus = []

    for tweet_content in tqdm(_tweet_contents, desc='학습 데이터 생성'):
        sentences_tag = okt.pos(tweet_content)

        noun_adj_list = []

        for word, tag in sentences_tag:
            if tag in ['Noun', 'Adjective'] and not common_utils.is_stop_words(word):
                noun_adj_list.append(word)

        sentence = ' '.join(noun_adj_list)

        # print(sentence)
        corpus.append(sentence)

    return corpus


def save_similar_words_to_csv(cosine_sim_result):
    df = DataFrame(cosine_sim_result)
    # print(df)
    df.to_csv(RESULT_COSINE_SIMILARITY_FILE, sep=',', index=False, header=['base_word', 'similar_words'])


if __name__ == '__main__':

    if 0 > ANALYSIS_DATA_RATIO or ANALYSIS_DATA_RATIO > 1:  # 잘못된 수치 값을 설정했을 때, 프로그램 종료
        logging.error(f'잘못된 ANALYSIS_DATA_RATIO 값: {ANALYSIS_DATA_RATIO}')
        exit(-1)

    tweet_data = pd.read_csv(TIME_LINE_DATA_FILE, encoding='cp949')

    tweet_contents = tweet_data[tweet_data['y'] == 1]['tweet'].fillna('')

    num_row = int(tweet_contents.shape[0] * ANALYSIS_DATA_RATIO)
    print(f'분석에 사용할 데이터 개수: {num_row}')

    tweet_contents = tweet_contents[:num_row+1]

    corpus = get_corpus(tweet_contents)

    tf_idf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')

    transpose_tfidf_matrix = tf_idf_vectorizer.fit_transform(corpus).T.toarray()

    cosine_sim = cosine_similarity(transpose_tfidf_matrix, transpose_tfidf_matrix)
    # print(cosine_sim)

    frequent_words = get_frequent_words()
    frequent_words_indices = get_words_indices(tf_idf_vectorizer, frequent_words)
    print(f'빈출 단어 개수 : {len(frequent_words_indices)}')

    cosine_sim_result = []
    for idx in frequent_words_indices:

        similar_words_result = []

        sim_scores = list(enumerate(cosine_sim[idx]))

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        word_indices = [i[0] for i in sim_scores[0:10]]

        # print(sim_scores)
        # print(word_indices)

        # 같은 단어가 들어갔으면 해당 단어를 제거하고, 다음 유사 단어 추가
        if idx in word_indices:
            skip_index = word_indices.index(idx)
            sim_scores.pop(skip_index)
            word_indices.remove(idx)

            index = sim_scores[10][0]
            word_indices.append(index)

        for i, word_index in enumerate(word_indices):
            similar_words_result.append((tf_idf_vectorizer.get_feature_names()[word_index], sim_scores[i][1]))

        cosine_sim_result.append((tf_idf_vectorizer.get_feature_names()[idx], similar_words_result))

        words = [tf_idf_vectorizer.get_feature_names()[i] for i in word_indices]
        # print(f'원래 단어 : {tf_idf_vectorizer.get_feature_names()[idx]}')
        # print(f'유사 단어 : {words}')

    # print(cosine_sim_result)

    save_similar_words_to_csv(cosine_sim_result)
