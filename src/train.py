import tensorflow as tf
import keras
import matplotlib as mpl
import PIL
import os
import os.path
import time
import shutil
from tqdm import tqdm
from keras.layers import Flatten, Dense, Input, Dropout
from keras_vggface.vggface import VGGFace
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

STUDENT_COUNT = 27
parentDir = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "tmp", "individualTraining"
)


def createTrainingDataset():
    os.makedirs(parentDir, exist_ok=True)
    trainingDir = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "tmp", "training"
    )
    for dir in tqdm(os.listdir(trainingDir)):
        if os.path.isfile(os.path.join(os.curdir, dir)):
            continue
        if dir == "unknown":
            continue
        os.makedirs(os.path.join(parentDir), exist_ok=True)
        os.makedirs(os.path.join(parentDir, dir, "unknown"), exist_ok=True)
        shutil.copytree(
            os.path.join(trainingDir, dir), os.path.join(parentDir, dir, dir)
        )
        imageCount = (
            len(os.listdir(os.path.join(parentDir, dir, dir)))
            - len(os.listdir(os.path.join(trainingDir, "unknown"))) // 2
        )
        for unknownPerson in os.listdir(trainingDir):
            if dir == unknownPerson:
                continue

            if os.path.isfile(os.path.join(os.curdir, unknownPerson)):
                continue

            count = max(imageCount // STUDENT_COUNT, 1)

            if unknownPerson == "unknown":
                count = len(os.listdir(os.path.join(trainingDir, "unknown"))) // 2

            for filename in os.listdir(os.path.join(trainingDir, unknownPerson)):
                if not filename.endswith(".jpg"):
                    continue

                if count <= 0:
                    break

                count -= 1

                shutil.copy(
                    os.path.join(trainingDir, unknownPerson, filename),
                    os.path.join(
                        parentDir,
                        dir,
                        "unknown",
                        f'{str(time.time()).replace(".", "_")}.jpg',
                    ),
                )


def buildCustomVGGMode(noOfStudents=56, hiddenDim=16):
    nbClass = noOfStudents + 1
    dataAugmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
            tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),
        ]
    )

    baseModel = VGGFace(include_top=False, input_shape=(224, 224, 3))
    baseModel.trainable = False

    inputs = Input(shape=(224, 224, 3))
    x = dataAugmentation(inputs)
    x = baseModel(inputs)
    x = Flatten(name="flatten")(x)
    x = Dense(hiddenDim, activation="relu", name="fc6")(x)
    out = Dense(nbClass, activation="softmax", name="classifier")(x)
    customVGGModel = keras.Model(inputs, out)
    return customVGGModel


def train():
    lr = 0.0001
    os.makedirs(
        os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "models"),
        exist_ok=True,
    )

    totalCount = 0
    for dir in os.listdir(parentDir):
        if not os.path.isfile(dir):
            totalCount += 1

    for i, dir in enumerate(os.listdir(parentDir)):
        if os.path.isfile(dir):
            continue

        print(f"Progress: {i+1}/{totalCount}")
        print("Training ", dir, "'s model")
        modelName = f"FaceRecognizer_{dir}_{int(time.time())}"
        tensorBoard = TensorBoard(log_dir=f"logs/{modelName}")
        earlyStopping = EarlyStopping(monitor="val_accuracy", patience=3)

        trainDS = keras.utils.image_dataset_from_directory(
            os.path.join(parentDir, dir),
            shuffle=True,
            batch_size=8,
            image_size=(224, 224),
            validation_split=0.2,
            subset="training",
            seed=1337,
        )
        validationDS = keras.utils.image_dataset_from_directory(
            os.path.join(parentDir, dir),
            shuffle=True,
            batch_size=8,
            image_size=(224, 224),
            validation_split=0.2,
            subset="validation",
            seed=1337,
        )
        model = buildCustomVGGMode(noOfStudents=1)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        model.fit(
            trainDS,
            epochs=10,
            validation_data=validationDS,
            callbacks=[tensorBoard, earlyStopping],
        )
        model.save(
            os.path.join(
                os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
                "models",
                f"{dir}_model.h5",
            ),
        )

def clean():
    shutil.rmtree(parentDir)


def main():
    createTrainingDataset()
    train()
    clean()


if __name__ == "__main__":
    main()
