# -*- encoding: utf-8 -*-
def is_stop_words(word: str):
    with open('./data/한국어불용어.txt', 'r') as file:
        stop_words = file.read().splitlines()

    if word in stop_words:
        return True
    else:
        return False


if __name__ == '__main__':
    print(is_stop_words('형호'))
