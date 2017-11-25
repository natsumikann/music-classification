# music-classification

## 実験の概要  

CNNを用いた楽曲分類を試みた。詳細はスライドにある通りです。３パターンのCNNを用いた実験と、LSTMを用いた実験を行い、４つそれぞれの実験のコードは下記のように別々のブランチに対応している。

## 実験と対応するブランチ
* master:このREADME.mdのみ
* jikken1:５層のCNNを利用し、batchnormalizationを入れた実験を行った。
* jikken2:6層のCNNを利用し、dropoutを入れた実験を行った。
* jikken3:ネットワークはjikken1と同じものを利用したが、波形データでなくメルスペクトログラムを学習させるようにした。
* lstm:CNNとの比較のため、LSTMをネットワークに採用して実験を行った。

## ファイルの説明  
* train_soundnet.py  
学習を行う。  
* test_soundnet.py  
テストを行う。  
* Regressor.py, Regressor_6layers.py  
ネットワークの定義がされている。  
* tag_dict.py  
もともと楽曲についている楽曲のジャンルのタグは様々な種類のものがあるのでpopかrockのどちらかに変換する。その変換のためのテーブルをCSVで出力する。  
* tag_dict.csv  
上のファイルを実行することで出力されたCSVファイル。これをtrainやtestで読み込んで楽曲のデータのタグを変換する。  
* Dataset_ta.py  
楽曲の5秒ずつ切り出してミニバッチ化し、train/testでデータとして利用する。  

## 依存ライブラリ
本コードは追加で以下のライブラリに依存する。
* mutagen 1.3.9
* PySoundFile 0.9.0
* LibROSA 0.5.1

## コードの実行方法(各ブランチ共通)
* 学習  
適当な場所にmusicディレクトリを用意。その下のディレクトリを（アーティスト名）＞（アルバム名）＞（曲名）のようにしてflac形式のtrain用楽曲データを配置する。`python train_soundnet.py -d <musicディレクトリのパス>`でファイルを実行。その他のオプションは`python train_soundnet.py -h`で参照。

* 分類  
musicディレクトリの下にtestディレクトリを用意し、その下に学習時と同様にディレクトリを作る。`python test_soundnet.py -d <testディレクトリのパス>`でファイルを実行。その他のオプションは`python test_soundnet.py -h`で参照。
