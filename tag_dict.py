# -*- coding: utf-8 -*-
# genre tagの一覧をcsvで出力
from mutagen.flac import FLAC
import glob
import csv
import os

def write_csv(path):
    files = glob.glob(os.path.join(path, '*/*/*.flac'))
    print(files)
    assert len(files) != 0
    audio = []
    with open('tag_dict.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for f in files:
            if ((FLAC(f).get('genre') not in audio) and (FLAC(f).get('genre') != None)):
                rows = []
                rows.append(FLAC(f).get('genre')[0])
                rows.append(' ')
                writer.writerow(rows)
                audio.append(FLAC(f).get('genre'))

    print(audio)


def count_data(path):
    files = glob.glob(os.path.join(path, '*/*/*.flac'))
    return len(files)


def read_csv():
    data = {}
    with open('tag_dict.csv', 'r') as f:
        reader = csv.reader(f, lineterminator='\n')
        for row in reader:
            data[row[0]] = row[1]
    return data

if __name__ == '__main__':
    write_csv("/music-tmp")
    print(count_data("/music-tmp"))