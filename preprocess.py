import glob
import h5py
import librosa
import numpy as np

def save(path, mel, cqt):
    f = h5py.File(path, 'w')
    f['mel'] = mel
    f['cqt'] = cqt
    f.close()

def main():
    res = glob.glob('data/lrm/*.mp3')
    num = 0
    for f in res:
        x, fs = librosa.load(f, sr=160000)
        mel = librosa.feature.melspectrogram(x, fs)
        cqt = librosa.feature.chroma_cqt(x, fs, n_chroma=24)
      
        mel = np.log(mel)
        
        length = mel.shape[1]
        k = length // 256
        for i in range(k):
             pmel = mel[:, i * 256: (i+1) * 256]
             pcqt = cqt[:, i * 256: (i+1) * 256]
             path = 'data/lrm_data/%03d_%03d.h5' % (num, i)
             print(path)
             save(path, pmel, pcqt)
        num += 1
             
        
        


if __name__ == '__main__':
    main()
