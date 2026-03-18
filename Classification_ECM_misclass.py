# Tensorflow Version 2.7 is needed
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from tensorflow import keras
import tensorflow as tf
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

##### Load EIS data-set #####

filename = "xy_data_16k_6circuit_v2.mat"

mat = scipy.io.loadmat(filename)

# raw EIS data: [Re(Z), Im(Z), freq]
x_raw = mat["x_data"]
y_raw = mat["y_data"]

y_raw = np.squeeze(y_raw)
x_raw = np.swapaxes(x_raw, 1, 2)

# one-hot labels for training
y_cat = tf.keras.utils.to_categorical(y_raw)

print("x_raw shape:", x_raw.shape)
print("y_raw shape:", y_raw.shape)

##### Data Augmentation #####
# Build augmented input for CNN only
new_shape = np.asarray(x_raw.shape)
new_shape[-1] = new_shape[-1] + 3
new_shape = tuple(new_shape)

new_x = np.zeros(new_shape)
new_x[:, :, :3] = x_raw

new_x[:, :, 3] = x_raw[:, :, 0] * -1
new_x[:, :, 4] = x_raw[:, :, 1] * -1
new_x[:, :, 5] = x_raw[:, :, 2] * -1

print("new_x shape:", new_x.shape)

##### Split data #####
# Split raw EIS and augmented EIS together so indices stay aligned
x_train_raw, x_test_raw, x_train, x_test, y_train, y_test = train_test_split(
    x_raw,
    new_x,
    y_cat,
    test_size=0.2,
    random_state=42
)

print("x_train_raw shape:", x_train_raw.shape)
print("x_test_raw shape:", x_test_raw.shape)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

##### Model #####
# drop rate 0.7
# batch size 1024

Experiment_name = "lab6basicECM_Classification_drop07_batch"
fn_tmp = filename.split("xy_data_", 1)[1].split(".", 1)[0]
Experiment_path = "EIS_" + fn_tmp + "_model_" + Experiment_name

os.makedirs(Experiment_path, exist_ok=True)

# build model
initializer = tf.keras.initializers.HeNormal()

def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1d = keras.layers.Conv1D(
        filters=64, kernel_size=32,
        padding="same", activation="relu",
        kernel_initializer=initializer
    )(input_layer)

    conv1d = keras.layers.Conv1D(
        filters=128, kernel_size=16,
        padding="same", activation="relu",
        kernel_initializer=initializer
    )(conv1d)

    conv1d = keras.layers.Conv1D(
        filters=256, kernel_size=8,
        padding="same", activation="relu",
        kernel_initializer=initializer
    )(conv1d)

    conv1d = keras.layers.Conv1D(
        filters=512, kernel_size=4,
        padding="same", activation="relu",
        kernel_initializer=initializer
    )(conv1d)

    conv1d = keras.layers.Conv1D(
        filters=768, kernel_size=2,
        padding="same", activation="relu",
        kernel_initializer=initializer
    )(conv1d)

    connector = keras.layers.SpatialDropout1D(0.7)(conv1d)
    connector = keras.layers.BatchNormalization()(connector)
    connector = keras.layers.GlobalAveragePooling1D()(connector)

    dense = keras.layers.Dense(
        1024,
        activation="relu",
        kernel_initializer=initializer
    )(connector)

    output_layer = keras.layers.Dense(6, activation="softmax")(dense)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

model = make_model(input_shape=x_train.shape[1:])
print(model.summary())

##### Training #####
epochs = 400
batch_size = 1024
Experiment_path = Experiment_path + "_" + str(batch_size)
os.makedirs(Experiment_path, exist_ok=True)

print("Experiment_path:", Experiment_path)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%y_%m_%d") + "/" \
          + Experiment_path.split("model_", 1)[1] \
          + "_" + filename.split("_", -1)[2] \
          + datetime.datetime.now().strftime("_%m%d%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=0,
    profile_batch=0
)

modelpath = Experiment_path + "/" + "model_{epoch:02d}_{val_loss:.2f}_{val_accuracy:.2f}.h5"

callbacks = [
    keras.callbacks.ModelCheckpoint(
        modelpath,
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=20,
        verbose=0,
        mode="min",
        min_lr=0.000001
    ),

    tensorboard_callback,
]

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=(x_test, y_test),
    verbose=2,
)

df_temp = pd.DataFrame(
    list(zip(
        history.history["accuracy"],
        history.history["val_accuracy"],
        history.history["loss"],
        history.history["val_loss"]
    )),
    columns=["accuracy", "val_accuracy", "loss", "val_loss"]
)
df_temp.to_csv(Experiment_path + "/" + "trainig_curve.csv", index=False)
print("Training finished.")

##### Evaluation #####

predict_model = model

x_t = x_test
y_t = y_test

m_ev = predict_model.evaluate(x_t, y_t, verbose=0)
y_pred = predict_model.predict(x_t, verbose=0)

# Use int labels
y_pred_class = np.zeros(len(y_t), dtype=int)
y_test_class = np.zeros(len(y_t), dtype=int)

for idx in range(len(y_t)):
    y_pred_class[idx] = np.argmax(y_pred[idx])

for idx in range(len(y_t)):
    y_test_class[idx] = np.argmax(y_t[idx])

test_list1 = y_test_class   # true
test_list2 = y_pred_class   # pred

cm = confusion_matrix(test_list1, test_list2)

##### Misclassification extraction #####

misclassified_idx = np.where(test_list1 != test_list2)[0]

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

##### Confusion matrix #####

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["C1", "C2", "C3", "C4", "C5", "C6"]
)

px = 1 / plt.rcParams["figure.dpi"]
fig, ax = plt.subplots(figsize=(600 * px, 600 * px), dpi=300)
disp.plot(cmap="summer", ax=ax)
cfmc_title = "Accuracy :" + str(m_ev[1] * 100) + "%" + "\n" + "Loss :" + str(m_ev[0])
plt.title(cfmc_title)
plt.savefig(Experiment_path + "/" + "CMatrix.png", bbox_inches="tight")
plt.close()

##### Count true labels #####

c1, c2, c3, c4, c5, c6 = 0, 0, 0, 0, 0, 0
for idx in range(len(test_list1)):
    if test_list1[idx] == 0:
        c1 += 1
    if test_list1[idx] == 1:
        c2 += 1
    if test_list1[idx] == 2:
        c3 += 1
    if test_list1[idx] == 3:
        c4 += 1
    if test_list1[idx] == 4:
        c5 += 1
    if test_list1[idx] == 5:
        c6 += 1

print("True label counts:", c1, c2, c3, c4, c5, c6)

##### Save misclassified RAW EIS only #####
# VERY IMPORTANT: save x_test_raw, not x_test

mis_data = []
mis_true = []
mis_pred = []
mis_index = []

for idx in misclassified_idx:
    mis_data.append(x_test_raw[idx])   # raw EIS: [Re, Im, freq]
    mis_true.append(int(test_list1[idx]))
    mis_pred.append(int(test_list2[idx]))
    mis_index.append(int(idx))

mis_data = np.array(mis_data)
mis_true = np.array(mis_true)
mis_pred = np.array(mis_pred)
mis_index = np.array(mis_index)

np.savez(
    Experiment_path + "/misclassified_samples.npz",
    data=mis_data,
    true_label=mis_true,
    pred_label=mis_pred,
    index=mis_index
)

print("Saved misclassified RAW EIS samples.")

##### Quick raw EIS plot check #####

def sort_eis(sample):
    """
    sample columns:
    [:,0] Re(Z)
    [:,1] Im(Z)
    [:,2] freq
    """
    order = np.argsort(sample[:, 2])[::-1]  # high freq -> low freq
    return sample[order]

def plot_eis(sample, true_label, pred_label):
    sample = sort_eis(sample)

    Zre = sample[:, 0]
    Zim = sample[:, 1]

    plt.figure(figsize=(4, 4))
    plt.plot(Zre, -Zim, 'o-')
    plt.xlabel("Z'")
    plt.ylabel("-Z''")
    plt.title(f"True C{true_label+1}  Pred C{pred_label+1}")
    plt.grid(True)
    plt.show()

print("Quick check for C2/C3 misclassified raw EIS:")
for i in mis_c2_c3[:5]:
    plot_eis(
        x_test_raw[i],
        int(test_list1[i]),
        int(test_list2[i])
    )

print("Quick check for C5/C6 misclassified raw EIS:")
for i in mis_c5_c6[:5]:
    plot_eis(
        x_test_raw[i],
        int(test_list1[i]),
        int(test_list2[i])
    )