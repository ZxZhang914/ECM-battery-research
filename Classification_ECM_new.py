#Tensorflow Version 2.7 is needed
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.io
from tensorflow import keras
import tensorflow as tf
import datetime
from numpy import unique
from numpy import argmax
from pandas import read_csv
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

##### Load EIS data-set #####

filename="xy_data_16k_6circuit_v2.mat"

x=scipy.io.loadmat(filename)["x_data"]
y=scipy.io.loadmat(filename)["y_data"]
y=np.squeeze(y)
x=np.swapaxes(x, 1, 2)
y=tf.keras.utils.to_categorical(y)


# Data Augmentation
new_shape=x.shape
new_shape=np. asarray(new_shape)
new_shape[-1]=new_shape[-1]+3
new_shape=tuple(new_shape)
new_x = np.zeros(new_shape)
new_x[:, :, :3] = x

new_x[:,:,3]=x[:,:,0]*-1
new_x[:,:,4]=x[:,:,1]*-1
new_x[:,:,5]=x[:,:,2]*-1

#Split data
x_train, x_test, y_train, y_test = train_test_split(new_x, y, test_size=0.2, random_state=42)

##### Model: 1D CNN + Flatten #####

Experiment_name="lab6basicECM_Classification_Conv1D"
fn_tmp=filename.split("xy_data_",1)[1].split(".",1)[0]
Experiment_path="EIS_"+fn_tmp+"_model_"+Experiment_name

DROPOUT = 0.3
LR = 1e-3

initializer = tf.keras.initializers.HeNormal()


def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    # 0. 输入归一化（按频率轴 z-score）
    x = keras.layers.LayerNormalization(axis=1, epsilon=1e-6)(input_layer)

    # 1. 1D Convolution blocks
    x = keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu",
                            kernel_initializer=initializer)(x)
    x = keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu",
                            kernel_initializer=initializer)(x)
    x = keras.layers.Conv1D(256, kernel_size=3, padding="same", activation="relu",
                            kernel_initializer=initializer)(x)

    # 2. Flatten
    x = keras.layers.Flatten()(x)

    # 3. 分类头
    x = keras.layers.Dense(512, activation="relu",
                            kernel_initializer=initializer)(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    x = keras.layers.Dense(256, activation="relu",
                            kernel_initializer=initializer)(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    output_layer = keras.layers.Dense(6, activation="softmax")(x)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=x_train.shape[1:])
#Model Summarize
print(model.summary())
#keras.utils.plot_model(model, show_shapes=True)

##### Training #####
epochs = 400
batch_size = 1024
Experiment_path=Experiment_path+"_"+str(batch_size)
print(Experiment_path)


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%y_%m_%d") + "/" \
                      + Experiment_path.split("model_",1)[1]  \
                      +"_"+ filename.split("_",-1)[2] \
                      + datetime.datetime.now().strftime("_%m%d%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                          log_dir=log_dir,
                                          histogram_freq=0,
                                          profile_batch=0)

modelpath= Experiment_path \
           + "/" + "model_{epoch:02d}_{val_loss:.2f}_{val_accuracy:.2f}.h5"

callbacks =[
            keras.callbacks.ModelCheckpoint(
                modelpath, save_best_only=True,
                monitor="val_loss",mode="min"
                ),

            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=20, verbose=0,
                mode='min', min_lr=0.000001
                ),

            # keras.callbacks.EarlyStopping(monitor="val_loss", patience=60,
            #                                verbose=1),

            #TqdmCallback(verbose=0),
            tensorboard_callback,
           ]

model.compile(
              optimizer=keras.optimizers.Adam(learning_rate=LR),
              loss="categorical_crossentropy",
              metrics=["accuracy"],
             )

history = model.fit(
          x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks,
          validation_data=(x_test,y_test),
          verbose=2,
                   )

df_temp = pd.DataFrame(list(zip(history.history["accuracy"],history.history["val_accuracy"],history.history["loss"],history.history["val_loss"])),
                            columns = ["accuracy","val_accuracy","loss","val_loss"])
df_temp.to_csv(Experiment_path+"/"+"trainig_curve.csv")
print(Experiment_path)

##### Evaluation #####

model_to_load = model

# summarize history for accuracy
fig=plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "validation"], loc="upper left")
plt.savefig(Experiment_path+"/"+"accuracy.png")
#plt.close(fig)
# plt.show()

# summarize history for loss
fig=plt.figure()
plt.plot(history.history["loss"][1:-1])
plt.plot(history.history["val_loss"][1:-1])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "validation"], loc="upper left")
plt.savefig(Experiment_path+"/"+"loss.png")
#plt.close(fig)
# plt.show()

#predict
predict_model = model

x_t= x_test
y_t= y_test
# x_t= x_train
# y_t= y_train

m_ev=predict_model.evaluate(x_t,y_t)
y_pred=predict_model.predict(x_t)
y_pred_class = np.zeros(len(y_t))
y_test_class = np.zeros(len(y_t))

for idx in range(len(y_t)):

    y_pred_class[idx]=np.argmax(y_pred[idx])

test_list2=y_pred_class

for idx in range(len(y_t)):

    y_test_class[idx]=np.argmax(y_t[idx])

test_list1=y_test_class

cm=confusion_matrix(test_list1,test_list2)

misclassified_idx = []

for i in range(len(test_list1)):
    if test_list1[i] != test_list2[i]:
        misclassified_idx.append(i)

print("Total misclassified:", len(misclassified_idx))

mis_c2_c3 = []
mis_c5_c6 = []

for i in misclassified_idx:

    true_label = int(test_list1[i])
    pred_label = int(test_list2[i])

    # C2 vs C3
    if (true_label == 1 and pred_label == 2) or (true_label == 2 and pred_label == 1):
        mis_c2_c3.append(i)

    # C5 vs C6
    if (true_label == 4 and pred_label == 5) or (true_label == 5 and pred_label == 4):
        mis_c5_c6.append(i)

print("C2/C3 misclassified:", len(mis_c2_c3))
print("C5/C6 misclassified:", len(mis_c5_c6))

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["C1","C2"
                              ,"C3","C4","C5", "C6"])
px = 1/plt.rcParams["figure.dpi"]  # pixel in inches
fig, ax = plt.subplots(figsize=(600*px, 600*px),dpi=300)
disp.plot(cmap="summer",ax=ax)
cfmc_title="Accuracy :"+str(m_ev[1]*100)+"%"+"\n"+"Loss :"+str(m_ev[0])
plt.title(cfmc_title)
plt.savefig(Experiment_path+"/"+"CMatrix.png")
# plt.show()

c1,c2,c3,c4,c5,c6=0,0,0,0,0,0
for idx in range(len(test_list1)):
    if test_list1[idx]==0:c1=c1+1
    if test_list1[idx]==1:c2=c2+1
    if test_list1[idx]==2:c3=c3+1
    if test_list1[idx]==3:c4=c4+1
    if test_list1[idx]==4:c5=c5+1
    if test_list1[idx]==5:c6=c6+1
print(c1,c2,c3,c4,c5,c6)

# save misclassified signals
mis_data = []
mis_true = []
mis_pred = []

for idx in misclassified_idx:

    mis_data.append(x_t[idx])
    mis_true.append(int(test_list1[idx]))
    mis_pred.append(int(test_list2[idx]))

mis_data = np.array(mis_data)
mis_true = np.array(mis_true)
mis_pred = np.array(mis_pred)

np.savez(
    Experiment_path+"/misclassified_samples.npz",
    data=mis_data,
    true_label=mis_true,
    pred_label=mis_pred,
    index=misclassified_idx
)

print("Saved misclassified samples")

def plot_eis(sample, true_label, pred_label):

    Zre = sample[:,0]
    Zim = sample[:,1]

    plt.figure(figsize=(4,4))
    plt.plot(Zre,-Zim,'o-')

    plt.xlabel("Z'")
    plt.ylabel("-Z''")
    plt.title(f"True C{true_label+1}  Pred C{pred_label+1}")
    plt.axis("equal")

    plt.show()

for i in mis_c2_c3[:5]:

    plot_eis(
        x_t[i],
        int(test_list1[i]),
        int(test_list2[i])
    )

for i in mis_c5_c6[:5]:

    plot_eis(
        x_t[i],
        int(test_list1[i]),
        int(test_list2[i])
    )
