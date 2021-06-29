import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)
from custom_callbacks import EarlyStopping, ModelCheckpoint
from data_helpers import DataGenerator
from seq2seq import Seq2Seq


INPUT_WIDTH = 21
LABEL_WIDTH = 15
LAYER_SIZE = 256
EPOCHS = 150
MIN_EPOCH = 100
BATCH_SIZE = 512
ATTENTION = True

RMSE = lambda x, y: mean_squared_error(x, y, squared=False)
MAE = mean_absolute_error
MAPE = mean_absolute_percentage_error

covid_fname = 'covid_20200301_20210619.csv'
model_fname = f'seq2seq_attn({ATTENTION})_iw{INPUT_WIDTH}_lw{LABEL_WIDTH}_ls{LAYER_SIZE}.h5'
feature_cols = ['decideDailyCnt', 'examCnt']
label_cols = 'decideDailyCnt'


def create_data_gen():
    covid_data = pd.read_csv(f'./data/{covid_fname}')
    data_gen = DataGenerator(raw_data=covid_data,
                             input_width=INPUT_WIDTH, label_width=LABEL_WIDTH,
                             feature_cols=feature_cols,
                             label_cols=label_cols,
                             norm=True, training=True,
                             train_split=1, val_split=0, test_split=0)
    return data_gen


def create_model_and_train(data_gen):
    train_callbacks = [EarlyStopping(min_epoch=MIN_EPOCH, patience=50, monitor='loss',
                                     verbose=1, restore_best_weights=True),
                       ModelCheckpoint(filepath=f'./models/{model_fname}', monitor='loss',
                            save_best_only=True, save_weights_only=True,
                            min_epoch=MIN_EPOCH, verbose=1)]

    seq2seq = Seq2Seq(units=LAYER_SIZE, 
                      input_width=INPUT_WIDTH, feature_num=len(feature_cols),
                      label_width=LABEL_WIDTH, attention=ATTENTION)
    history = seq2seq.train(data_gen.dataset.batch(BATCH_SIZE), 
                            # val_ds=data_gen.val.batch(BATCH_SIZE),
                            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
                            callbacks=train_callbacks)
    return seq2seq, history


def evaluation(y_true, y_pred, methods):
    '''
        Args:
            y_true (numpy.array): (batch, LABEL_WIDTH) 형태의 실제 값
            y_pred (numpy.array): (batch, LABEL_WIDTH) 형태의 예측 값
            methods (list): 평가에 사용할 메소드 리스트

        Returns:
            results (numpy.array): (batch, len(methods)) 형태의 성능 평가 값

        methods 리스트 내부에 사용될 method는 기본적으로 리스트 형태의 x, y를 받아 결과를 출력하는 형태의 함수
    '''
    results = [[method(x, y) for method in methods] for x, y in zip(y_true, y_pred)]
    results = np.array(results)
    results = np.mean(results, axis=0)
    results = np.around(results, 2)
    return  results


if __name__=='__main__':
    data_gen = create_data_gen()
    seq2seq, history = create_model_and_train(data_gen)

    os.makedirs('./images', exist_ok=True)

    plt.plot(history.history['loss'], label='loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig(f'./images/model_history.png')
    plt.close()

    # seq2seq = Seq2Seq(units=LAYER_SIZE, 
    #                   input_width=INPUT_WIDTH, feature_num=len(feature_cols),
    #                   label_width=LABEL_WIDTH, attention=ATTENTION)
    # seq2seq.model.load_weights(f'./models/{model_fname}')

    for inp, targ in data_gen.dataset.batch(data_gen.data_size):
        preds = seq2seq.predict(inp)
    
    plt.plot(tf.concat([targ[:, 0], targ[-1, 1:]], axis=0))
    for i, pred in enumerate(preds):
        plt.plot(range(i, i+LABEL_WIDTH), pred)
    plt.savefig('./images/model_predict_value(test).png')
    plt.close()

    plt.plot(targ[:, 0], label='targ')
    plt.plot(preds[:, 0], label='preds')
    plt.legend()
    plt.savefig('./images/model_predict(test_step).png')
    plt.close()

    plt.plot(targ[:, LABEL_WIDTH-1], label='targ')
    plt.plot(preds[:, LABEL_WIDTH-1], label='preds')
    plt.legend()
    plt.savefig(f'./images/model_predict(test_step{LABEL_WIDTH-1}).png')
    plt.close()

    with open(f'./data/{covid_fname[:-4]}_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print(evaluation(scaler.inverse_transform(targ),
                     scaler.inverse_transform(preds),
                     methods=[RMSE, MAE, MAPE]))    
    print('done')
