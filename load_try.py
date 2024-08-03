import NN_final as nn
import librosa
from datagen import Datagen  
from audioPreprocessing import *
sampleSize=9000

dic={0:'inhale',
     1:'exhale',
     2:'none'}


audio='audio/20210218_221917.mp3'

model=nn.Model.load('100_ep.sv')

assert audioFile(audio),'not an audio file'

signals,X=Datagen.onRunFullProcess(audio,sampleSize)

for signal,x  in zip(signals,X):
    confidence=model.predict(x)
    prediction=model.output_layer_activation.prediction(confidence)
    print(signal,dic[prediction[0]])

y,sr=noise(audio)
librosa.display.waveshow(y=y,sr=sr)
plt.show()
