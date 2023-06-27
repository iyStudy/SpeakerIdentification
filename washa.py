import scipy.io.wavfile as wav # .wavファイルを扱うためのライブラリ
from sklearn.svm import SVC     # SVC（クラス分類をする手法）を使うためのライブラリ
import numpy                    # ndarray（多次元配列）などを扱うためのライブラリ
import librosa                  # 音声信号処理をするためのライブラリ
import os                       # osに依存する機能を利用するためのライブラリ
import pyaudio
import wave

import speech_recognition as sr
import subprocess
import tempfile
import speech_recognition

RECORD_SECONDS = 5          # 録音する時間:秒
iDeviceIndex = 0            # 録音デバイスの番号 PC:0 プロジェクタ:1
FILENAME = 'inPut.wav'      # 保存するファイル名
FORMAT = pyaudio.paInt16    # 音声フォーマット
CHANNELS = 1                # チャンネル数（モノラル）
RATE = 44100                # サンプリングのレート
CHUNK = 2**11               # データ点数

################　↓ 音声識別部分 ↓　########################

# ルートディレクトリ
ROOT_PATH = '/Users/a71378/Desktop/gijoroku/'
TRAGET_PATH = ROOT_PATH + 'work/'

# 話者の名前（各話者のデータのディレクトリ名になっている）
speakers=['ishida','ID02','ID03','OTHER']

word_training=[]    # 学習用のMFCCの値を格納する配列
speaker_training=[] # 学習用のラベルを格納する配列

# MFCCを求める関数
def getMfcc(filename):
    y, sr = librosa.load(filename)          # 引数で受けとったファイル名でデータを読み込む。
    return librosa.feature.mfcc(y=y, sr=sr) # MFCCの値を返します。

# 各ディレクトリごとにデータをロードし、MFCCを求めていく
for speaker in speakers:
    # どの話者のデータを読み込んでいるかを表示
    print('Reading data of %s...' % speaker)
    # 話者名でディレクトリを作成しているため<ルートパス+話者名>で読み込める。
    path = os.path.join(ROOT_PATH + speaker)    
    # パス、ディレクトリ名、ファイル名に分けることができる便利なメソッド
    for pathname, dirnames, filenames in os.walk(path): 
        for filename in filenames:
            # macの場合は勝手に.DS_Storeやらを作るので、念の為.wavファイルしか読み込まないように
            if filename.endswith('.wav'):
                print('Reading data of .wav')
                mfcc=getMfcc(os.path.join(pathname, filename))

                word_training.append(mfcc.T)    # word_trainingにmfccの値を追加(配列を転置)
                label=numpy.full((mfcc.shape[1] ,), # mfcc.shape[1]はmfccの数
                speakers.index(speaker), dtype=numpy.int)   # labelをspeakersのindexで全て初期化
                speaker_training.append(label)  # speaker_trainingにラベルを追加

word_training=numpy.concatenate(word_training)  # ndarrayを結合
speaker_training=numpy.concatenate(speaker_training)


##### 機械学習
# カーネル係数を1e-4で学習
clf = SVC(C=1, gamma=1e-4)      # SVCはクラス分類をするためのメソッド
clf.fit(word_training, speaker_training)    # MFCCの値とラベルを組み合わせて学習
print('Learning Done')


counts = []     # predictionの中で各値（予測される話者のインデックス）が何回出ているかのカウント
file_list = []  # file名を格納する配列

################ ↑　音声識別部分 ↑　########################



#### 録音してwavファイルを作成する関数  ここから　######
print("議事録を開始します！")
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    input_device_index=iDeviceIndex,
                    frames_per_buffer=CHUNK)
print("recording...")       # 録音開始
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("finish recording")   # 録音終了

stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(TRAGET_PATH + FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

#### 録音してwavファイルを作成する関数  ここまで　######


# workフォルダに保存した録音ファイルを参照
path = os.path.join(ROOT_PATH + 'work')
print("path:" + path) #
for pathname, dirnames, filenames in os.walk(path):
    
    for filename in filenames:
        if filename.endswith('.wav'):
            #print("何を見てるか確認 filename:" + filename) #
            mfcc = getMfcc(os.path.join(pathname, filename))
            prediction = clf.predict(mfcc.T)    # MFCCの値から予測した結果を代入
            # predictionの中で各値（予測される話者のインデックス）が何回出ているかをカウントして追加
            counts.append(numpy.bincount(prediction))   

            testprint3 = str(prediction)
            # print("確認する prediction:" + testprint3) #
            file_list.append(filename)  # 実際のファイル名を追加

# 推測される話者の名前の抽出
for filename, count in zip(file_list, counts): 
    result = speakers[numpy.argmax(count-count.mean(axis=0))] 




f = open('gijiroku.txt','a')
print("議事録を記入します！")
# 音声入力
r = sr.Recognizer()


try:
    # Google Web Speech APIで音声認識
    #text = r.recognize_google(audio, language="ja-JP")
    with speech_recognition.AudioFile(TRAGET_PATH+FILENAME) as src:
        audio = r.record(src)
    text = r.recognize_google(audio, language="ja-JP")
except sr.UnknownValueError:
    print("Google Web Speech APIは音声を認識できませんでした。")
except sr.RequestError as e:
    print("GoogleWeb Speech APIに音声認識を要求できませんでした;"
            " {0}".format(e))
#else:
if  text == "終了":
    print(text)
    #break
else:
    print("発言者:"+result)
    print(text)
    f.write(text + '\n') # 内容を書き込み
print("完了。")
f.close()


