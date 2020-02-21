import numpy as np
import matplotlib.pylab as plt
import sklearn.datasets as skdata
import sklearn

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)

data = imagenes.reshape((n_imagenes, -1))
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.7)

def distancia(x_train,y_train):
    numero = 1
    dd = y_train==numero
    cov = np.cov(x_train[dd].T)
    valores, vectores = np.linalg.eig(cov)
    valores = np.real(valores)
    vectores = np.real(vectores)
    ii = np.argsort(-valores)
    valores = valores[ii]
    vectores = vectores[:,ii]
    
    #Predicción de la mínima distancia
    vect = vectores[:,6]
    x_train1=x_train[dd]
    dist=[]
    for i in x_train1:
        dist.append(np.linalg.norm(np.dot(vect,i)))
    dist=np.array(dist)
    dist=np.mean(dist)
    
    return [vect,dist]
def predict(imagen,x_train,y_train):
    d=distancia(x_train,y_train)[1]
    vect=distancia(x_train,y_train)[0]
    uno=0
    d1=np.linalg.norm(np.dot(vect,imagen))
    if(d1<d):
        uno=1
    return uno

y_train1=y_train
index=np.where(y_train1!=1)
y_train1[index]=0

y_test1=y_test
index=np.where(y_test1!=1)
y_test1[index]=0

predict1=[]
for i in x_test:
    predict1.append(predict(i,x_train,y_train1))
predict1=np.array(predict1)

predict2=[]
for i in x_train:
    predict2.append(predict(i,x_train,y_train1))
predict2=np.array(predict2)


F1=sklearn.metrics.f1_score(y_test1, predict1)
F12=sklearn.metrics.f1_score(y_train1, predict2)

plt.figure()
plt.subplot(121)
plt.imshow(sklearn.metrics.confusion_matrix(y_test1, predict1))
plt.title("F1 para Test = {:.2f}".format(F1))
plt.subplot(122)
plt.imshow(sklearn.metrics.confusion_matrix(y_train1, predict2))
plt.title("F1 para Train = {:.2f}".format(F12))
plt.savefig('matriz_de confusión.png')