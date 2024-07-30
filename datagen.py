import librosa
import numpy as np
from audioPreprocessing import *
import matplotlib.pylab as plt
import soundfile as sf

def dataPointGen(path_sr,path_des):
    labels=['inhale','exhale','none']
    print(f"0:{labels[0]}\n1:{labels[1]}\n2:{labels[2]}\n3:skip this file")
    os.makedirs(path_des,exist_ok=True)
    for i in labels:
        os.makedirs(os.path.join(path_des,i),exist_ok=True)
    files=audioFileCheck(path_sr)
    for file in files:
        print(f'file: {file}')
        path=os.path.join(path_sr,file)
        y,sr=librosa.load(path)

        new_y=y/np.abs(y[np.argmax(np.abs(y))])
        new_y_p=np.insert(new_y,0,0)
        new_y=np.insert(new_y,len(new_y)-1,0)
        gradient=(new_y-new_y_p)/sr
        # print(f'inhale start timestamp: {np.argmax(gradient)/sr}')
        inhale=np.argmax(gradient)
        index=inhale
        def endv(index):
            while(True):
                if(gradient[index]==0):
                    return index
                index+=1

        signal=[]
        gInd=0
        while(gInd<len(gradient)):
            if gradient[gInd]>gradient[inhale]/8:
                end=endv(gInd)
                signal.append((gradient[gInd],gInd/sr,end/sr))
                gInd=end
            else:
                gInd+=1

        print("signals detected:\n",signal)
        librosa.display.waveshow(y=new_y,sr=sr)
        plt.show()
        for i in signal:
            a=int(input(f'what is {i}'))
            if a==3: break
            if a not in [0,1,2]:
                print('Try whole thing again')
                os.rmdir(path_des)
            startT=int(i[1]*sr)
            endT=int(i[2]*sr)
            sf.write(os.path.join(path_des,labels[a],file),y[startT:endT],sr)



        # print(f'inhale end timestamp: {index/sr}')
        # print(f'hold duration:{np.average([signal[1][1],signal[1][2]])-np.average([signal[0][1],signal[0][2]])}')

    
dataPointGen(path_sr='cleaned_audio',path_des='dataset') #put your paths here
        

