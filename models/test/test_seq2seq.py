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
from utils.dataset import DataSlicer
from utils.collect import get_nation_covid_data
from models import seq2seq


INPUT_WIDTH = 14
LABEL_WIDTH = 15
LAYER_SIZE = 512
DROPOUT = 0.2
EPOCHS = 200
MIN_EPOCH = 50
BATCH_SIZE = 256
ATTENTION = True

RMSE = lambda x, y: mean_squared_error(x, y, squared=False)
MAE = mean_absolute_error
MAPE = mean_absolute_percentage_error

data_path = "./data"
saved_path = './saved'
image_path = './images/seq2seq'

start_date = "20200303"
end_date = "20210805"

covid_fname = f"covid_{start_date}_{end_date}.csv"
scaler_fname = f"covid_{start_date}_{end_date}_scaler.pkl"
model_fname = f"seq2seq_iw{INPUT_WIDTH}_lw{LABEL_WIDTH}_ls{LAYER_SIZE}_do{DROPOUT}_diff{2}.h5"
feature_cols = ["decideCnt"]
label_cols = ["decideCnt"]


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
    print("Load data...")
    csv_path = f"{data_path}/{covid_fname}"
    scaler_path = f"{data_path}/{scaler_fname}"

    if os.path.exists(csv_path):
        covid_data = pd.read_csv(csv_path)
    else:
        print("Request API...")
        covid_data = get_nation_covid_data(start_date, end_date)
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    target_input_data = np.diff(covid_data[feature_cols], n=2, axis=0)
    target_label_data = np.diff(covid_data[label_cols], n=2, axis=0)

    data_slicer = DataSlicer(input_data=scaler.transform(target_input_data),
                            label_data=scaler.transform(target_label_data),
                            input_width=INPUT_WIDTH, label_width=LABEL_WIDTH,
                            teacher_force=True)

    seq2seq_model = seq2seq.get_model(LAYER_SIZE, INPUT_WIDTH, len(feature_cols),
                                      LABEL_WIDTH, len(label_cols), DROPOUT, ATTENTION)
    history = seq2seq_model.fit(data_slicer.train.batch(BATCH_SIZE),
                                validation_data=data_slicer.val.batch(BATCH_SIZE),
                                epochs=EPOCHS, callbacks=[
                                    EarlyStopping(min_epoch=EPOCHS//4, patience=EPOCHS//10),
                                    ModelCheckpoint(f"{saved_path}/seq2seq_iw{INPUT_WIDTH}_lw{LABEL_WIDTH}_ls{LAYER_SIZE}_do{DROPOUT}_diff{2}.h5",
                                                    save_best_only=True, save_weights_only=True, min_epoch=EPOCHS//4)
                                ])
    
    os.makedirs(image_path, exist_ok=True)

    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.savefig(f"{image_path}/seq2seq_history.png")
    plt.close()

    seq2seq_model.load_weights(f"{saved_path}/seq2seq_iw{INPUT_WIDTH}_lw{LABEL_WIDTH}_ls{LAYER_SIZE}_do{DROPOUT}_diff{2}.h5")

    total_inp, total_targ = list(data_slicer.total.batch(data_slicer.data_size))[0]

    enc_out, enc_state = seq2seq_model.layers[2](total_inp[0])
    total_prds = tf.cast(total_inp[1][:, :1], dtype=tf.float32)
    for step in range(LABEL_WIDTH):
        prd = seq2seq_model.layers[3](total_prds, enc_state, enc_out)
        total_prds = tf.concat([total_prds, prd[:, -1:, tf.newaxis]], axis=1)
    total_prds = tf.squeeze(total_prds[:, 1:], axis=-1)
    total_targ = tf.squeeze(total_targ, axis=-1)

    targ = scaler.inverse_transform(total_targ)[-data_slicer.test_size:]
    preds = scaler.inverse_transform(total_prds)[-data_slicer.test_size:]

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

    print(evaluation(targ,
                     preds,
                     methods=[RMSE, MAE, MAPE]))    
    print("done")
