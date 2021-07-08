import tensorflow as tf


class EarlyStopping(tf.keras.callbacks.EarlyStopping):
    """ keras의 EarlyStopping을 상속받음. 

    특정 반복 횟수 이후부터 monitor를 시작함.

    Attributes:
        min_epoch: (int) 해당 콜백 함수가 동작하기 시작할 최소 epoch
    """
    def __init__(self, monitor="val_loss", min_delta=0, patience=0, verbose=0, mode="auto", min_epoch=0, baseline=None, restore_best_weights=False):
        super(EarlyStopping, self).__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode, baseline=baseline, restore_best_weights=restore_best_weights)
        self.min_epoch = min_epoch
    
    def on_epoch_end(self, epoch, logs):
        if epoch < self.min_epoch:
            return
        return super().on_epoch_end(epoch, logs=logs)


class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """ keras의 ModelCheckpint을 상속받음. 

    특정 반복 횟수 이후부터 monitor를 시작함.

    Attributes:
        min_epoch: (int) 해당 콜백 함수가 동작하기 시작할 최소 epoch
    """
    def __init__(self, filepath, monitor="val_loss", verbose=0, min_epoch=0, save_best_only=False, save_weights_only=False, mode="auto", save_freq="epoch", options=None, **kwargs):
        super().__init__(filepath, monitor=monitor, verbose=verbose, save_best_only=save_best_only, save_weights_only=save_weights_only, mode=mode, save_freq=save_freq, options=options, **kwargs)
        self.min_epoch = min_epoch
    
    def on_epoch_end(self, epoch, logs):
        if epoch < self.min_epoch:
            return
        return super().on_epoch_end(epoch, logs=logs)
