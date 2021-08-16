import time
from random import uniform

import twint
import pandas as pd
from os import mkdir, path
from datetime import timedelta, datetime
import re


# Valid Directory Naming Function
def clean_name(term):
    cleaned = re.sub('[^0-9a-zA-Zㄱ-ㅎ가-힣]', '', term)  # 특수문자 제거
    return cleaned


# Twint collect User tweets
def twint_user_tweets(dirname, user, since, until, filename, limit):
    c = twint.Config()

    c.Limit = limit
    c.Username = user
    c.Since = since
    c.Until = until
    c.Hide_output = True
    # c.Store_json = True
    c.Store_csv = True
    c.Output = filename
    c.Debug = True
    c.Resume = f'{dirname}/save_endpoint/save_endpoint_{since}.txt'


    try:
        twint.run.Search(c)

    except (KeyboardInterrupt, SystemExit):
        raise

    except:
        print(f"Problem with {since}.")


# Twint Search Function
def twint_search(dirname, searchterm, since, until, json_name, limit):
    c = twint.Config()

    # Parameter setting
    c.Limit = limit
    c.Search = searchterm
    c.Since = since
    c.Until = until
    c.Hide_output = True
    # c.Store_json = True
    c.Store_csv = True
    c.Output = json_name
    c.Debug = True
    c.Resume = f'{dirname}/save_endpoint/save_endpoint_{since}.txt'
    c.Popular_tweets = False

    try:
        twint.run.Search(c)

    except (KeyboardInterrupt, SystemExit):
        raise

    except:
        print(f"Problem with {since}.")


# Loop Function, Default number of tweet is 50
def twint_loop(searchterm, since, until, limit=50, is_user=False):
    dirname = clean_name(searchterm)

    # Create target directory
    try:
        mkdir(dirname)
        mkdir(f'{dirname}/save_endpoint')
        print("Directory", dirname, "Created ")
    except FileExistsError:
        print("Directory", dirname, "already exists")

    # Loop
    daterange = pd.date_range(since, until)

    for start_date in daterange:
        since = start_date.strftime("%Y-%m-%d")
        until = (start_date + timedelta(days=1)).strftime("%Y-%m-%d")

        json_name = "".join(since.split("-")) + ".csv"
        json_name = path.join(dirname, json_name)

        print(f'Getting {since} ')
        if is_user:
            twint_user_tweets(dirname, searchterm, since, until, json_name, limit)
        else:
            twint_search(dirname, searchterm, since, until, json_name, limit)


if __name__ == "__main__":

    # arrUsers = ['hann7336', 'HtruWruHzvJ5kKE', 'M6hTGEtsy19KhNo']
    #
    # for user in arrUsers:
    #     twint_loop(user, '2021-04-01', '2021-05-07', limit=1000, is_user=True)
    #     f = uniform(1.6, 3.5)
    #     time.sleep(f)
    #     print(f'Sleep {f} second')
    # quit('DONE')

    ## trends
    arrTendKeyword = ['피어리스', '호흡곤란', '어버이날', '미세먼지', '보건실 침대', '허벅지 씨름', '수업시간 몰폰', '더블헤더']
    for Keyword in arrTendKeyword:
        twint_loop(Keyword, '2021-05-01', '2021-05-07', limit=1000)
        f = uniform(1.6, 3.5)
        time.sleep(f)
        print(f'Sleep {f} second')
    quit('DONE')

    ##act words
    arrTypeKeyword00 = ["###성매매 관련 게시글을 찾는 검색어: 성매매 행위###"]
    for Keyword in arrTypeKeyword00:
        twint_loop(Keyword, '2021-04-07', '2021-05-07', limit=1000)
        f = uniform(0.6, 3.5)
        time.sleep(f)
        print(f'Sleep {f} second')

    quit('DONE')
    ## typeA
    arrTypeKeyword00 = ["###성매매 관련 게시글을 찾는 검색어: 성매매 유형A###"]
    for Keyword in arrTypeKeyword00:
        twint_loop(Keyword, '2021-04-07', '2021-05-07', limit=1000)
        f = uniform(0.6, 3.5)
        time.sleep(f)
        print(f'Sleep {f} second')

    ## typeB
    arrTypeKeyword00 = ["###성매매 관련 게시글을 찾는 검색어: 성매매 유형B###"]
    for Keyword in arrTypeKeyword00:
        twint_loop(Keyword, '2020-04-07', '2021-05-07', limit=1000)
        f = uniform(-1.6, 3.5)
        time.sleep(f)
        print(f'Sleep {f} second')
