import librosa
import numpy as np
from audioPreprocessing import *
import matplotlib.pylab as plt
import soundfile as sf

class Datagen():
    def __init__(self,path_sr,path_des,labels=['inhale','exhale','none']) -> None:
        self.path_sr=path_sr
        self.path_des=path_des
        self.labels=labels

    def getData(self,file):

        def endv(index):
            while(True):
                if(gradient[index]==0):
                    return index
                index+=1

        print(f'file: {file}')
        path=os.path.join(self.path_sr,file)
        y,sr=librosa.load(path)

        new_y=y/np.abs(y[np.argmax(np.abs(y))])
        new_y_p=np.insert(new_y,0,0)
        new_y=np.insert(new_y,len(new_y)-1,0)
        gradient=(new_y-new_y_p)/(1/sr)
        inhale=np.argmax(gradient)

        signal=[]
        gInd=0

        while(gInd<len(gradient)):
            if gradient[gInd]>gradient[inhale]*0.01: #1% rate change of the strongest signal
                end=endv(gInd)
                if((end-gInd)/sr>0.4):
                    signal.append((gradient[gInd],gInd/sr,end/sr))
                gInd=end
            else:
                gInd+=1

        print(f'No of signals detected: {len(signal)}')
        print("signals detected:\n",signal)
        librosa.display.waveshow(y=new_y,sr=sr)
        plt.show()

        for i in signal:
                a=int(input(f'what is {i}'))
                if a==3: break
                if a not in [0,1,2]:
                    print('Try whole thing again')
                    os.rmdir(self.path_des)
                startT=int(i[1]*sr)
                endT=int(i[2]*sr)
                sf.write(os.path.join(self.path_des,self.labels[a],file),y[startT:endT],sr)
            

    def dataPointGen(self):

        print(f"0:{self.labels[0]}\n1:{self.labels[1]}\n2:{self.labels[2]}\n3:skip this file")
        os.makedirs(self.path_des,exist_ok=True)
        for i in self.labels:
            os.makedirs(os.path.join(self.path_des,i),exist_ok=True)
        files=audioFileCheck(self.path_sr)
        for file in files:
            self.getData(file)
    
#method names are weird because i am bad at naming things this the best i can comeup with
#sample dry run replace your paths here
demo=Datagen(path_sr='cleaned_audio',path_des='dataset')  #if you have different labels 
                                                          #pass in labels as a list under the parameter 
                                                          #name labels
        
demo.dataPointGen()

"""
Note:(IMP)
listen after the plot shows you decide which signal is which 0,1 or 2 if you want to skip the file all
together press 3 after you have closed the plot window it will ask signal by signal which is which so before 
you close the plot make your mind or you can also use audio or video side by side then it will move on to
the next signal one the whole audio is sorted it will move on to the next one in the folder
if you don't get it ask me
and for (from audioPreprocessing import *) to work you need to have the audioPreprocesssing file in the same
folder
"""
