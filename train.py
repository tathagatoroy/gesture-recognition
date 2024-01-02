import os
import tqdm
import random
import os
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import gesture_recognizer

import matplotlib.pyplot as plt


if __name__=="__main__":
    dataset_path = "/scratch/tathagato/hagrid_dataset_512"
    #dataset_path = "/scratch/tathagato/new/hagrid_dataset_512"
    data = gesture_recognizer.Dataset.from_folder(
    dirname=dataset_path,
    hparams=gesture_recognizer.HandDataPreprocessingParams()
)   
    print("done with data processing")
    train_data, rest_data = data.split(0.8)
    validation_data, test_data = rest_data.split(0.5)

    hparams = gesture_recognizer.HParams(export_dir="exported_model")
    options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
    print(hparams)
    print(options)
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options
    )

    loss, acc = model.evaluate(test_data, batch_size=1)
    print(f"Test loss:{loss}, Test accuracy:{acc}")
    model.export_model()
    print("done with training")
