import librosa 
import os
import noisereduce as nr
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

extentions=['.mp3','.wav']


def normalize(y):
    max=np.abs(y[np.argmax(np.abs(y))])
    if max==1:
        return y
    return y/np.abs(y[np.argmax(np.abs(y))])

def audioFile(path):
    _,extention=os.path.splitext(path)
    if extention in extentions:
        return True
    return False

def audioFileCheck(path):
    dirlist=[]
    for i in os.listdir(path):
        if audioFile(i):
            dirlist.append(i)
    return dirlist

def noise(file,path_des=None):

    y,sr=librosa.load(file)
    reduce_noise=nr.reduce_noise(sr=sr,y=y)
    if path_des==None:
        return reduce_noise,sr
    outputfile=os.path.join(path_des,f'{os.path.splitext(os.path.basename(file))[0]}_cleaned.wav')
    sf.write(outputfile,reduce_noise,sr)

def noise_cancellation(path_sr,path_des):

    os.makedirs(path_des,exist_ok=True)    
    if os.path.isfile(path_sr):
        noise(path_sr,path_des)
    elif os.path.isdir(path_sr):
        for i in audioFileCheck(path_sr):
            path=os.path.join(path_sr,i)
            if os.path.isfile(path):
                noise(path,path_des)
    else:
        return 1
    
    return 0

def bestPair(num):
    pairList=[]
    div=num
    while(div>0):
        if(num%div==0):
            pairList.append([div,int(num/div)])
        div-=1
    for _ in range(int(len(pairList)/2)):
        pairList.pop()
    return pairList[np.argmin(np.sum(pairList,axis=1))]


def plot(path):

    plt.figure(figsize=(10,3))
    fileList=audioFileCheck(path)
    assert len(fileList), "no file found"
    X,Y=bestPair(len(fileList))
    fig,axs = plt.subplots(X,Y,figsize=(12,8))
    fileIndex=0
    for x in range(X):
        for y in range(Y):
            file=os.path.join(path,fileList[fileIndex])
            amp, sr = librosa.load(file,sr=None)
            axs[x][y].set_title(fileList[fileIndex])
            librosa.display.waveshow(amp,sr=sr,ax=axs[x][y])
            fileIndex+=1
    plt.tight_layout()
    plt.show()

def resize(sampleSize,y,sr):

    new_sr=int((sampleSize*sr)/len(y))
    y=normalize(y)
    y_resample=librosa.resample(y,orig_sr=sr,target_sr=new_sr)
    if len(y_resample)<sampleSize:
        padding=sampleSize-len(y_resample)
        y_resample=np.concatenate((y_resample,np.zeros((padding))))
    return [y_resample]


