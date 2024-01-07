from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import rgb_to_grayscale
from tensorflow.image import resize
from matplotlib import pyplot
from os import listdir
import pandas as pd
import numpy as np
import random as rd




def charger_chemins(chem_mura, type_dataset):  
    """On récupère les chemins d'accès aux images depuis le chemin d'accès au dataset MURA
    type_dataset permet de choisir entre les données d'entraînements et de validation"""
    
    tab_chem = pd.read_csv(chem_mura+"/{}_labeled_studies.csv".format(type_dataset))
    tab_chem = np.array(tab_chem)
    
    return tab_chem



def charger_img(chemin):
    """Fonction permettant de charger une image à partir de son chemin d'accès en 
    la transformant en niveaux de gris et en l'adaptant à un format standard"""
    
    img = load_img(chemin)
    img = resize(img, (300,300))
    img = img_to_array(img)
    img_gs = rgb_to_grayscale(img)
    return img_gs.numpy()




def charger_dataset(membre,tab_chem, longueur=100000):
    """Charge le dataset correspondant à une certaine partie du corps
    à partir des chemins données sous forme de liste ou d'array"""
    
    dataset, label = [], []
    
    i=0         #Pour mon pc de fragile
    
    for elem in tab_chem:
        dossier = elem[0]
        if not(membre in dossier):
            continue
        
        if i > longueur: break         #longueur du dataset pour les test
    
        for img in listdir(dossier):
            dataset.append(charger_img(dossier+img))
            label.append(elem[1]) 
            i+=1
            print(i)
   
    dataset = np.asarray(dataset)
    label = np.asarray(label)
    
    return (dataset,label)
    
def charger_dataset_partiel(membre,tab_chem, longueur, plage):
    """Charge le dataset correspondant à une certaine partie du corps
    à partir des chemins données sous forme de liste ou d'array
    Il faut préciser dans quelles lignes du fichier excel on peut trouver les images correspondantes"""
    
    dataset, label = [], []
    
    indices = rd.sample(range(plage[0], plage[1]), longueur)
    
    for i in range(longueur):
        elem = tab_chem[indices[i]]
        dossier = elem[0]
        if not(membre in dossier):
            i-=1
            continue
    
        for img in listdir(dossier):
            dataset.append(charger_img(dossier+img))
            label.append(elem[1])
            print(i)
   
    dataset = np.asarray(dataset)
    label = np.asarray(label)
    
    return (dataset,label)

    
def define_model(ker_size):
    """Modèle CNN utilisé pour la prédiction des anomalies"""
    
    model = Sequential()
    model.add(Conv2D(32, ker_size, activation='relu', input_shape=(300, 300, 1)))     
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, ker_size, activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model    
    
  
def prep_model(dataset, label, b_s, valX, valY, nb_e =5, ker_size = (5,5)):
    
    model = define_model(ker_size)
    history = model.fit(dataset, label, validation_data=(valX,valY), batch_size=b_s, epochs=nb_e)
    
    return model, history
    
    

# normalize images
def prep_normalize(train, test):
    """Les pixels sont linéairement distribués dans [0,1]"""
	# convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
	# return normalized images
    return train_norm, test_norm


 
# standardize images
def prep_standardize(train, test):
    """On fixe la moyenne à 0 et l'écart-type à 1 en considérant l'ensemble
    du dataset d'entraînement"""
    
	# convert from integers to floats
    train_stan = train.astype('float32')
    test_stan = test.astype('float32')
    # calculate statistics
    m = train_stan.mean()
    s = train_stan.std()
    # center datasets
    train_stan = (train_stan - m) / s
    test_stan = (test_stan - m) / s
    # return normalized images
    return train_stan, test_stan


# repeated evaluation of model with data prep scheme
def repeated_evaluation(train_ds, train_label, v_ds, v_lab, ker_size, batch_size = 8, n_repeats=10):

    # repeated evaluation
    scores = list()
    for i in range(n_repeats):
        model,_ = prep_model(train_ds, train_label, batch_size, v_ds, v_lab)
        _, acc = model.evaluate(v_ds, v_lab, verbose=0)
        scores.append(acc)
        print('> %d: %.3f' % (i, acc * 100.0))
    return scores


#Exécution du programme


print("GO")
#train_ds, train_label = charger_dataset_partiel("SHOULDER", charger_chemins("MURA-v1.1", "train"), 600,(2,2600))        #Données d'entraînement
#v_ds, v_label = charger_dataset_partiel("SHOULDER", charger_chemins("MURA-v1.1", "valid"), 100, (680,860))                 #Données de validation

train_ds, train_label = charger_dataset_partiel("WRIST", charger_chemins("MURA-v1.1", "train"), 500, (7110,10550))        #Données d'entraînement
v_ds, v_label = charger_dataset("WRIST", charger_chemins("MURA-v1.1", "valid"))

#pre-processing
train_ds, v_ds = prep_normalize(train_ds, v_ds)


print("Go pour l'entraînement")
model, history = prep_model(train_ds, train_label, 8, v_ds, v_label, 6, (5,5))




# evaluate the model
_, train_acc = model.evaluate(train_ds, train_label)
_, test_acc = model.evaluate(v_ds, v_label)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()



""" batch_size

scores = []
for i in range(4,32, 4):
    model,_ = prep_model(train_ds, train_label, i, v_ds, v_lab, 6)
    _, acc = model.evaluate(v_ds, v_lab, verbose=0)
    scores.append(acc)
    print('> %d: %.3f' % (i, acc * 100.0))

"""










