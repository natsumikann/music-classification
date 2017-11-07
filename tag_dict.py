# -*- coding: utf-8 -*-
# genre tagの一覧をcsvで出力
from mutagen.flac import FLAC
import glob
import csv

path = '/music/'

files = glob.glob(path + '*/*/*.flac')
print(files)
assert len(files) != 0
audio = []
with open('tag_dict.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for f in files:
        if ((FLAC(f).get('genre') not in audio) and (FLAC(f).get('genre') != None)):
            writer.writerow(FLAC(f).get('genre'))
            audio.append(FLAC(f).get('genre'))

print(audio)
