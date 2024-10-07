import librosa
import numpy as np
import matplotlib.pylab as plt
import soundfile as sf
import audioPreprocessing
import moviepy.editor as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
import os
import platform
import subprocess
import cv2
import pickle

class Datagen():
    def __init__(self,path_sr=None,path_des=None,labels=['inhale','exhale','mouthpiece','tabletDrop','none']) -> None:
        self.path_sr=path_sr
        self.path_des=path_des
        self.labels=labels
        self.processed=[]
        self.load()

    @staticmethod
    def onRunFullProcess(path,sampleSize):
        signal,y,sr=Datagen.detectSignal(path)
        x=[]
        for s in signal:
            x.append(audioPreprocessing.resize(sampleSize,y[int(s[1]*sr):int(s[2]*sr)],sr))
        
        return signal,x

    @staticmethod
    def endv(gradient,index):
        while(True):
            if(np.abs(gradient[index])<1e-5):
                return index
            index+=1

    @staticmethod
    def detectSignal(path):

        if os.path.splitext(path)[1]=='.mp4':
            video=mp.VideoFileClip(path)
            video.audio.write_audiofile('temp.mp3')
            y,sr=audioPreprocessing.noise('temp.mp3')
            os.remove('temp.mp3') 
        else:
            y,sr=audioPreprocessing.noise(path)        
        new_y=audioPreprocessing.normalize(y)
        new_y_p=np.insert(new_y,0,0)
        new_y_z=np.insert(new_y,len(new_y)-1,0)
        gradient=(new_y_z-new_y_p)/(1/sr)
        inhale=np.argmax(gradient)

        signal=[]
        gInd=0

        while(gInd<len(gradient)):
            if gradient[gInd]>gradient[inhale]*0.01: #1% rate change of the strongest signal
                end=Datagen.endv(gradient,gInd)
                if((end-gInd)/sr>0.4):
                    signal.append((gradient[gInd],gInd/sr,end/sr))
                gInd=end
            else:
                gInd+=1
        
        return signal,new_y,sr
    
    @staticmethod
    def saveVideo(inputPath,outputPath,start,end):

        video = VideoFileClip(inputPath)

        cap = cv2.VideoCapture(inputPath)

        if not cap.isOpened():
            print("Error: Could not open video.")
        else:
            # Get the original width and height of the video
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release

        cropped_video = video.subclip(start, end)

        cropped_video = cropped_video.resize(newsize=(original_width, original_height))

        cropped_video.write_videofile(outputPath,codec="libx264")

        video.close()
        cropped_video.close()

    def getData(self,file,label_data=True):

        print(f'file: {file}')
        if self.path_sr!=None:
            path=os.path.join(self.path_sr,file)
        else:
            path=file
        signal,y,sr=self.detectSignal(path)


        print(f'No of signals detected: {len(signal)}')
        print("signals detected:\n",signal)

        Platform=platform.system()
        if (Platform=='Darwin'):
            print(f'opening file {path}')
            subprocess.call(('open',path))
        else:
            os.startfile(path)

        if label_data:
            for ind,i in enumerate(signal):
                librosa.display.waveshow(y=y,sr=sr)
                print(f'what is {i}')
                plt.show()
                a=int(input())
                if a==5: continue
                elif a==6: break
                elif a==7: return False          
                if a not in [0,1,2,3,4]:
                    print('Try whole thing again')
                    os.rmdir(self.path_des)
                file_name=os.path.splitext(file)[0].split('/')[-1]+f'_{ind}'+'.mp4'
                if a in [0,1,2,3]:
                    self.saveVideo(path,os.path.join(self.path_des,"video",self.labels[a],file_name),i[1],i[2])
                startT=int(i[1]*sr)
                endT=int(i[2]*sr)
                file_name=os.path.splitext(file)[0].split('/')[-1]+f'_{ind}'+'.mp3'
                sf.write(os.path.join(self.path_des,"audio",self.labels[a],file_name),y[startT:endT],sr)
        return True
    
    def load(self):
        if not os.path.exists("test.fsv"):
            return None
        with open("test.fsv", 'rb') as file:
            self.processed = pickle.load(file)
    
    def save(self):
        with open("test.fsv", 'wb') as file:
            pickle.dump(self.processed, file)

    def dataPointGen(self):

        print(f"0:{self.labels[0]}\n1:{self.labels[1]}\n2:{self.labels[2]}\n3:{self.labels[3]}\n4:{self.labels[4]}\n5:skip this signal\n6:skip this file\n7:save")
        os.makedirs(self.path_des,exist_ok=True)
        os.makedirs(os.path.join(self.path_des,"audio"),exist_ok=True)
        os.makedirs(os.path.join(self.path_des,"video"),exist_ok=True)
        for i in self.labels:
            os.makedirs(os.path.join(self.path_des,"audio",i),exist_ok=True)
            if i in ['inhale','exhale','mouthpiece','tabletDrop']:
                os.makedirs(os.path.join(self.path_des,"video",i),exist_ok=True)
        files=audioPreprocessing.audioFileCheck(self.path_sr)
        ns=0
        for file in files:
            if file in self.processed:
                continue
            elif not self.getData(file):
                ns=1
                self.save()
                break
            else:
                self.processed.append(file)
        if ns==0:
            if os.path.exists("test.fsv"):
                os.remove("test.fsv")



    
#method names are weird because i am bad at naming things this the best i can comeup with
#sample dry run replace your paths here
if __name__=='__main__':
    demo=Datagen(path_sr='test',path_des='testData')  #if you have different labels 
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
