import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    ''' 인코더
    번역 과정(Machine Translation)에서는 source language를 입력으로 받아 특징을 추출
    시계열 예측에서는 t 이전 시점의 데이터를 입력으로 받고 특징을 추출  

    RNN으로 GRU를 사용함
    '''
    def __init__(self, units, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        self.units = units
        self.gru = tf.keras.layers.GRU(units, 
                                       return_sequences=True, 
                                       return_state=True)
    
    def call(self, inputs, training=False):
        outputs, state = self.gru(inputs)
        return outputs, state

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({'class_name': 'Encoder', 'config': {'units': self.units}})
        return config


class Decoder(tf.keras.layers.Layer):
    ''' 디코더
    Translation에서는 시작 토큰(e.g., <SOS>)과 인코더의 벡터를 받아 단어를 예측하고, 
    예측한 단어를 다시 입력으로 사용해 연속적으로 예측함. 

    시계열 예측에서는 t시점의 데이터를 시작 토큰으로 주고 이후 n-step을 예측함
    인코더와 동일하게 RNN으로 GRU를 사용    

    Attention의 경우 디코더의 state를 계산할 때 이전 state와 input만을 참고하는 것이 아니라
    인코더의 모든 step에서의 출력을 추가로 참고하여 state를 계산
    다양한 종류의 Attention이 있으며 여기서는 dot-product attention을 사용
    '''
    def __init__(self, units, attention=False, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)
        self.units = units
        self.attention = attention

        self.gru = tf.keras.layers.GRU(units, 
                                       return_sequences=True,
                                       return_state=True)
        if attention:
            self.attn = tf.keras.layers.Attention()
        self.dense = tf.keras.layers.Dense(1, activation='relu')

    def call(self, inputs, hidden, enc_output, training=False):
        # inputs => (batch, label_width)
        # hidden => (batch, enc_dim)
        # enc_output => (batch, input_width, enc_dim)

        outputs = None
        attn_weights = None
        sub_input = inputs[:, tf.newaxis, tf.newaxis, 0] # (batch, 1, 1)
        state = hidden
        # 학습 과정일 경우 teacher forcing을 사용함
        for t in range(inputs.shape[1]):
            if self.attention:
                # context_vec => (batch, 1, enc_dim)
                context_vec, attn_weight = self.attn([state[:, tf.newaxis, :,], enc_output],
                                                     return_attention_scores=True)
                # sub_input과 context_vec을 연결함. (attetnion 적용)
                # sub_input => (batch, 1, 1+enc_dim)
                sub_input = tf.concat([sub_input, context_vec], axis=-1)
            
            # output => (batch, 1, dec_dim)
            output, state = self.gru(sub_input, initial_state=state) 

            # (batch, 1, dec_dim) => (batch, 1, 1)
            output = self.dense(output)

            # output이 None이 아닐 경우
            if outputs is not None:
                # (batch, "here concat", dec_dim) step을 이어붙임
                outputs = tf.concat([outputs, output], axis=1)
            else:
                # None이면 초기 값으로 할당
                outputs = output

            if self.attention:
                if attn_weights is not None:
                    # (batch, "here concat", dec_dim) step을 이어붙임
                    attn_weights = tf.concat([attn_weights, attn_weight], axis=1)
                else:
                    # None이면 초기 값으로 할당
                    attn_weights = attn_weight

            # 마지막 step일 때 t+1로 인해 index error가 발생하므로 pass
            if t < inputs.shape[1]-1:
                if training:
                    # 학습 시, 다음 실제 값을 decoder의 입력으로 사용
                    sub_input = inputs[:, tf.newaxis, tf.newaxis, t+1]
                else:
                    # 예측 시, 예측 값을 다음 입력으로 사용
                    sub_input = output
        
        # (batch, predict_step)
        outputs = tf.squeeze(outputs)
        self.attn_wegiths = attn_weights

        return outputs

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({'class_name': 'Decoder', 'config': {'units': self.units}})
        return config



class Seq2Seq():
    def __init__(self, units, input_width, feature_num, label_width,
                 attention=False):
        self.units = units
        self.input_width = input_width
        self.feature_num = feature_num
        self.label_width = label_width
        self.attention = attention

        self._build()

    def _build(self):
        self.enc_input = tf.keras.Input(shape=(self.input_width, self.feature_num))
        self.encoder = Encoder(self.units)
        self.dec_input = tf.keras.Input(shape=(self.label_width))
        self.decoder = Decoder(self.units, attention=self.attention)

        enc_outputs, enc_state = self.encoder(self.enc_input)
        dec_outputs = self.decoder(self.dec_input, enc_state, enc_outputs)

        self.model = tf.keras.Model(inputs=(self.enc_input, self.dec_input),
                                    outputs=dec_outputs)
        self.model.compile(loss='mse', optimizer='adam')

    def train(self, train_ds, val_ds=None, epochs=1, batch_size=None,
              verbose='auto', callbacks=None, **kwargs):
        history = self.model.fit(train_ds, validation_data=val_ds, 
                                 epochs=epochs, batch_size=batch_size, verbose=verbose,
                                 callbacks=callbacks, **kwargs)
        return history

    def predict(self, inputs):
        return self.model(inputs, training=False)
