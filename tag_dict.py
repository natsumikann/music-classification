# -*- coding: utf-8 -*-
# genre tagの一覧をcsvで出力
from mutagen.flac import FLAC
import glob
import csv
import os
import sys

def write_csv(path):
    '''
    与えられたPATH内の、有効なTagが付いたFLACファイルからすべての"genre"タグの一覧を生成
    :param path: 音楽ファイルの格納されたトップディレクトリ アルバムアーティスト/アルバム/ファイル の階層
    :return: None
    '''
    files = glob.glob(os.path.join(path, '*/*/*.flac'))
    # print(files)
    assert len(files) != 0
    print(len(files), 'Files')
    audio = []
    valid_files = 0
    with open('tag_dict.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for f in files:
            if ((FLAC(f).get('genre') != None)):
                valid_files += 1
        for f in files:
            if (FLAC(f).get('genre') not in audio) and (FLAC(f).get('genre') is not None):
                rows = []
                rows.append(FLAC(f).get('genre')[0])
                rows.append(' ')
                writer.writerow(rows)
                audio.append(FLAC(f).get('genre'))
    print(valid_files, 'Valid Files')
    print(audio)


def search_tag(path, tag):
    '''
    与えられた"genre"タグの値がどのファイルに属するかの一覧を表示
    :param path: 音楽ファイルの格納されたトップディレクトリ アルバムアーティスト/アルバム/ファイル の階層
    :param tag: 検索対象の"genre"タグ
    :return: None
    '''
    files = glob.glob(os.path.join(path, '*/*/*.flac'))
    # print(files)
    assert len(files) != 0
    print(len(files), 'Files')
    for f in files:
        if FLAC(f).get('genre') is not None:
            if tag == FLAC(f).get('genre')[0]:
                print(f)



def count_data(path):
    '''
    与えられたディレクトリ以下の音楽ファイルの総数を返す。
    :param path: 音楽ファイルの格納されたトップディレクトリ アルバムアーティスト/アルバム/ファイル の階層
    :return: 与えられたディレクトリ以下の音楽ファイルの総数
    '''
    files = glob.glob(os.path.join(path, '*/*/*.flac'))
    return len(files)


def read_csv():
    '''
    write_csv()関数で生成されたCSVファイルを編集し各"genre"タグと"pop", "rock"の対応付けを行ったものをパース
    :return: "genre"タグをキーとし"pop"または"rock" をvalueとする辞書
    '''
    data = {}
    with open('tag_dict.csv', 'r') as f:
        reader = csv.reader(f, lineterminator='\n')
        for row in reader:
            data[row[0]] = row[1]
    return data

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Wrong usage")
    if sys.argv[1] == '-w':
        if len(sys.argv) != 3:
            print("Wrong usage")
        write_csv(sys.argv[2])
    elif sys.argv[1] == '-s':
        if len(sys.argv) != 4:
            print("Wrong usage")
        search_tag(sys.argv[2], sys.argv[3])
    else:
        print(read_csv())
