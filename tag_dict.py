# -*- coding: utf-8 -*-
# genre tagの一覧をcsvで出力
from mutagen.flac import FLAC
import glob
import csv

path = '/music/'

def write_csv():
    files = glob.glob(path + '*/*/*.flac')
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

def read_csv():
    data = {}
    with open('tag_dict.csv', 'r') as f:
        reader = csv.reader(f, lineterminator='\n')
        for row in reader:
            data[row[0]] = row[1]
    return data

if __name__ == '__main__':
    print(read_csv())