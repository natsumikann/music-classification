# -*- coding: utf-8 -*-
# genre tagの一覧をcsvで出力
from mutagen.flac import FLAC
import glob
import csv
import os
import sys

def write_csv(path):
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
    files = glob.glob(os.path.join(path, '*/*/*.flac'))
    # print(files)
    assert len(files) != 0
    print(len(files), 'Files')
    for f in files:
        if FLAC(f).get('genre') is not None:
            if tag == FLAC(f).get('genre')[0]:
                print(f)



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