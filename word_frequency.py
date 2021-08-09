# -*- encoding: utf-8 -*-
import pandas as pd
from wordcloud import WordCloud
from konlpy.tag import Okt
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import common_utils
import logging
from pandas import DataFrame

TIME_LINE_DATA_FILE = './data/트윗크롤링(이상치 처리).csv'

RESULT_WORD_COUNT_CSV_FILE = './data/word-frequency-result-100.csv'
RESULT_WORD_CLOUD_IMAGE_FILE = './data/word-frequency-result-100.png'


ANALYSIS_DATA_RATIO = 0.01  # 분석에 사용될 데이터 비율 선택 (데이터 범위: 0~1)

okt = Okt()


def get_word_counts(_word_list, num=100):
    counts = Counter(_word_list)
    word_counts = counts.most_common(num)

    return word_counts


def save_word_counts_to_csv(_word_counts):
    df = DataFrame(_word_counts)
    df.to_csv(RESULT_WORD_COUNT_CSV_FILE, sep=',', index=False, header=['word', 'count'])


def visualize_word_cloud(_word_counts):
    wc = WordCloud(font_path='./font/GodoM.ttf', background_color="white", width=1000, height=1000, max_font_size=300)
    word_cloud = wc.generate_from_frequencies(dict(_word_counts))
    # 생성된 이미지 저장 맟 사각화
    word_cloud.to_file(RESULT_WORD_CLOUD_IMAGE_FILE)
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    plt.imshow(word_cloud)
    plt.show()


def get_noun_adj_words(_tweet_contents):
    noun_adj_list = []
    for idx, tweet_content in enumerate(tqdm(_tweet_contents)):
        sentence_tags = okt.pos(tweet_content)

        # tag가 명사이거나 형용사인 단어들만 noun_adj_list 삽입
        for word, tag in sentence_tags:
            if tag in ['Noun', 'Adjective'] and not common_utils.is_stop_words(word):
                noun_adj_list.append(word)

    return noun_adj_list


if __name__ == '__main__':

    if 0 > ANALYSIS_DATA_RATIO or ANALYSIS_DATA_RATIO > 1:  # 잘못된 수치 값을 설정했을 때, 프로그램 종료
        logging.error(f'잘못된 ANALYSIS_DATA_RATIO 값: {ANALYSIS_DATA_RATIO}')
        exit(-1)

    tweet_data = pd.read_csv(TIME_LINE_DATA_FILE, encoding='cp949')

    tweet_contents = tweet_data[tweet_data['y'] == 1]['tweet'].fillna('')  # 트윗 내용만 가져옴

    num_row = int(tweet_contents.shape[0] * ANALYSIS_DATA_RATIO)
    print(f'분석에 사용할 데이터 개수: {num_row}')

    tweet_contents = tweet_contents[:num_row+1]

    word_list = get_noun_adj_words(tweet_contents)

    # 빈출 단어 100개 추출
    word_counts = get_word_counts(word_list, 100)

    save_word_counts_to_csv(word_counts)

    visualize_word_cloud(word_counts)
