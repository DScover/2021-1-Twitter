# -*- encoding: utf-8 -*-
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import logging
from konlpy.tag import Okt
import common_utils
import networkx as nx
import operator
import matplotlib.pyplot as plt
import platform


TIME_LINE_DATA_FILE = './data/트윗크롤링(이상치 처리).csv'
SEARCH_RESULT_DATA_FILE = './data/계정크롤링(전처리-일반,타광고삭제) (계정정보,타임라인 트윗 모두 ' \
                          '포함).csv'

WORD_PAIR_CSV_FILE = './data/networkx.csv'

RESULT_NETWORK_IMAGE_FILE = './data/network-result(모든_데이터_사용).png'

MAX_NODE_NUM = 100
EDGE_THRESHOLD = 1

ANALYSIS_DATA_RATIO = 0.01  # 분석에 사용될 데이터 비율 선택 (데이터 범위: 0~1)


okt = Okt()


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


if __name__ == '__main__':

    if 0 > ANALYSIS_DATA_RATIO or ANALYSIS_DATA_RATIO > 1:  # 잘못된 수치 값을 설정했을 때, 프로그램 종료
        logging.error(f'잘못된 ANALYSIS_DATA_RATIO 값: {ANALYSIS_DATA_RATIO}')
        exit(-1)

    tweet_data1 = pd.read_csv(TIME_LINE_DATA_FILE, encoding='cp949')
    tweet_data2 = pd.read_csv(SEARCH_RESULT_DATA_FILE, encoding='cp949')

    tweet_contents = tweet_data1[tweet_data1['y'] == 1]['tweet'].dropna()

    tweet_contents = tweet_contents.append(tweet_data2['tweet'].dropna())

    tweet_contents = tweet_contents.sample(frac=1, random_state=0).reset_index(drop=True)

    num_row = int(tweet_contents.shape[0] * ANALYSIS_DATA_RATIO)
    print(f'분석에 사용할 데이터 개수: {num_row}')

    tweet_contents = tweet_contents[:num_row+1]

    corpus = get_corpus(tweet_contents)

    tf_idf_vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')

    tfidf_matrix = tf_idf_vectorizer.fit_transform(corpus).toarray()

    column_list = tf_idf_vectorizer.get_feature_names()
    word_length = len(column_list)

    count_dict = {}
    for doc_number in tqdm(range(num_row+1), desc='동시 출현 빈도 계산'):
        tmp = tweet_contents[doc_number]
        for i, word1 in enumerate(column_list):
            if word1 in tmp:
                for j in range(i + 1, word_length):
                    if column_list[j] in tmp:
                        count_dict[column_list[i], column_list[j]] = \
                            count_dict.get((column_list[i], column_list[j]), 0) \
                            + max(tfidf_matrix[doc_number][i], tfidf_matrix[doc_number][j])

    # count_list에 word1, word2, frequency 형태로 저장
    count_list = []
    for i, words in enumerate(count_dict):
        count_list.append([words[0], words[1], count_dict[words]])

    # 단어쌍 동시 출현 빈도는 DataFrame 형식
    df = pd.DataFrame(count_list, columns=["word1", "word2", "freq"])
    df = df.sort_values(by=['freq'], ascending=False)
    df = df.reset_index(drop=True)

    # 단어쌍 동시 출현 빈도 결과 파일 저장
    df.to_csv(WORD_PAIR_CSV_FILE, encoding='utf-8-sig')

    dataset = pd.read_csv(WORD_PAIR_CSV_FILE)
    dataset = dataset[:MAX_NODE_NUM+1]

    G_centrality = nx.Graph()
    # edge 연결
    for ind in range((len(np.where(dataset['freq'] >= EDGE_THRESHOLD)[0]))):
        G_centrality.add_edge(dataset['word1'][ind], dataset['word2'][ind], weight=int(dataset['freq'][ind]))

    # 중심성 게산
    dgr = nx.degree_centrality(G_centrality)
    pgr = nx.pagerank(G_centrality)

    sorted_dgr = sorted(dgr.items(), key=operator.itemgetter(1), reverse=True)
    sorted_pgr = sorted(pgr.items(), key=operator.itemgetter(1), reverse=True)

    G = nx.Graph()
    # 중심성에 따라 노드의 크기 결정
    for i in range(len(sorted_pgr)):
        G.add_node(sorted_pgr[i][0], nodesize=sorted_dgr[i][1])

    for idx in range((len(np.where(dataset['freq'] >= EDGE_THRESHOLD)[0]))):
        G.add_weighted_edges_from([(dataset['word1'][idx], dataset['word2'][idx], int(dataset['freq'][idx]))])

    # node 크기 조정
    sizes = [G.nodes[node]['nodesize'] * 1000 for node in G]

    # edge 크기 조정
    widths = list(nx.get_edge_attributes(G, 'weight').values())
    norm_widths = [0.2 + (float(i)-min(widths))/(max(widths)-min(widths)) * 4 for i in widths]

    options = {
        'edge_color': '#FFDEA2',
        'with_labels': True,
        'font_weight': 'regular',
        'width': norm_widths,
        'font_size': 8
    }

    if platform.system() == 'Darwin':  # 맥
        nx.draw(G, node_size=sizes, pos=nx.spring_layout(G, k=3.5, iterations=100), **options,
                font_family='AppleGothic')
    elif platform.system() == 'Windows':  # 윈도우
        nx.draw(G, node_size=sizes, pos=nx.spring_layout(G, k=3.5, iterations=100), **options,
                font_family='Malgun Gothic')

    ax = plt.gca()

    plt.axis('off')
    plt.savefig(RESULT_NETWORK_IMAGE_FILE, dpi=300, format='PNG')
    plt.show()
