import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)
from utils.callbacks import EarlyStopping, ModelCheckpoint
from utils.dataset import DataGenerator
from utils.collect import get_nation_covid_data
from models.seq2seq import Seq2Seq


INPUT_WIDTH = 21
LABEL_WIDTH = 15
LAYER_SIZE = 256
EPOCHS = 100
MIN_EPOCH = 50
BATCH_SIZE = 128
ATTENTION = True

RMSE = lambda x, y: mean_squared_error(x, y, squared=False)
MAE = mean_absolute_error
MAPE = mean_absolute_percentage_error

data_path = "./data"
saved_path = './saved'
image_path = './images/seq2seq'

start_date = "20200303"
end_date = "20210717"

covid_fname = f"covid_{start_date}_{end_date}.csv"
model_fname = f"seq2seq_attn({ATTENTION})_iw{INPUT_WIDTH}_lw{LABEL_WIDTH}_ls{LAYER_SIZE}.h5"
feature_cols = ["decideCnt"]
label_cols = "decideCnt"


def create_data_gen(csv_path):
    """ Data Generator 생성 함수

    Args:
        data_path (str): 전체 csv 파일의 경로. utils.collect 참고.
    
    Returns:
        data_gen (class): 데이터 셋이 포함된 DataGenerator 
    """
    print("Load data...")
    if os.path.exists(csv_path):
        covid_data = pd.read_csv(csv_path)
    else:
        print("Request API...")
        covid_data = get_nation_covid_data(start_date, end_date)
        
    data_gen = DataGenerator(raw_data=covid_data,
                             input_width=INPUT_WIDTH, label_width=LABEL_WIDTH,
                             feature_cols=feature_cols,
                             label_cols=label_cols,
                             norm=True, training=True,
                             train_split=0.6, val_split=0.15, test_split=0.25)
    return data_gen


def create_model_and_train(data_gen):
    """ 모델 생성 및 학습 함수

    Args:
        data_gen (class): 데이터 셋이 포함된 DataGenerator

    Returns:
        seq2seq (class): Encoder-Decoder 구조의 Seq2Seq 모델
        history (class): 학습 정보를 담고 있는 객체

    Examples:
        >>> seq2seq, history = create_model_and_train(data_gen)
        >>> plt.plot(history.history['loss'])
        >>> plt.show()
        >>> seq2seq.predict(data_gen.test)
        <tf.tensor: shape=(...) ... >
    """
    train_callbacks = [EarlyStopping(min_epoch=MIN_EPOCH, patience=50, 
                                     verbose=1, restore_best_weights=True),
                       ModelCheckpoint(filepath=f"{saved_path}/{model_fname}",
                            save_best_only=True, save_weights_only=True,
                            min_epoch=MIN_EPOCH, verbose=1)]

    seq2seq = Seq2Seq(units=LAYER_SIZE, 
                      input_width=INPUT_WIDTH, feature_num=len(feature_cols),
                      label_width=LABEL_WIDTH, attention=ATTENTION)
    history = seq2seq.train(data_gen.train.batch(BATCH_SIZE), 
                            val_ds=data_gen.val.batch(BATCH_SIZE),
                            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
                            callbacks=train_callbacks)
    return seq2seq, history


def evaluation(y_true, y_pred, methods):
    """ 모델이 예측한 값을 통해 모델을 평가하는 함수

        Args:
            y_true (numpy.array): (batch, LABEL_WIDTH) 형태의 실제 값
            y_pred (numpy.array): (batch, LABEL_WIDTH) 형태의 예측 값
            methods (list): 평가에 사용할 메소드 리스트. methods 리스트 내부에 사용될 method는 기본적으로 리스트 형태의 x, y를 받아 결과를 출력하는 형태의 함수

        Returns:
            results (numpy.array): (batch, len(methods)) 형태의 성능 평가 값. 각 batch마다 methods를 사용한 성능의 평균을 반환

        Examples:
            >>> abs_diff = lambda x, y: abs(x-y)
            >>> evaluation(y_true, y_pred, [abs_diff])
    """
    results = [[method(x, y) for method in methods] for x, y in zip(y_true, y_pred)]
    results = np.array(results)
    results = np.mean(results, axis=0)
    results = np.around(results, 2)
    return  results


if __name__=="__main__":
    data_gen = create_data_gen(f"{data_path}/{covid_fname}")
    seq2seq, history = create_model_and_train(data_gen)

    os.makedirs(image_path, exist_ok=True)

    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.savefig(f"{image_path}/seq2seq_history.png")
    plt.close()

    seq2seq = Seq2Seq(units=LAYER_SIZE, 
                      input_width=INPUT_WIDTH, feature_num=len(feature_cols),
                      label_width=LABEL_WIDTH, attention=ATTENTION)
    seq2seq.model.load_weights(f"{saved_path}/{model_fname}")

    for inp, targ in data_gen.dataset.batch(data_gen.data_size):
        preds = seq2seq.predict(inp)

    with open(f"{data_path}/{covid_fname[:-4]}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    targ = scaler.inverse_transform(targ)[-data_gen.test_size:]
    preds = scaler.inverse_transform(preds)[-data_gen.test_size:]

    plt.plot(tf.concat([targ[:, 0], targ[-1, 1:]], axis=0))
    for i, pred in enumerate(preds):
        plt.plot(range(i, i+LABEL_WIDTH), pred)
    plt.savefig(f"{image_path}/seq2seq_all_step.png")
    plt.close()

    plt.plot(targ[:, 0], label="targ")
    plt.plot(preds[:, 0], label="preds")
    plt.legend()
    plt.savefig(f"{image_path}/seq2seq_0_step.png")
    plt.close()

    plt.plot(targ[:, LABEL_WIDTH-1], label="targ")
    plt.plot(preds[:, LABEL_WIDTH-1], label="preds")
    plt.legend()
    plt.savefig(f"{image_path}/seq2seq_{LABEL_WIDTH-1}_step.png")
    plt.close()

    print(evaluation(scaler.inverse_transform(targ),
                     scaler.inverse_transform(preds),
                     methods=[RMSE, MAE, MAPE]))    
    print("done")
