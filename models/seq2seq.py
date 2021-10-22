import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    """ Seq2Seq 모델의 Encoder layer

    Translation에서는 source language를 입력으로 받아 특징을 추출

    시계열 예측에서는 t 이전 시점의 데이터를 입력으로 받고 특징을 추출  

    GRU를 사용함

    Attributes:
        units: (int) GRU의 차원
        dropout: (float) dropout
    """
    def __init__(self, units, dropout, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        self.units = units
        self.gru = tf.keras.layers.GRU(units, 
                                       return_sequences=True, 
                                       return_state=True, dropout=dropout)
    
    def call(self, inputs, training=False):
        outputs, state = self.gru(inputs)
        return outputs, state

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"class_name": "Encoder", "config": {"units": self.units}})
        return config


class Decoder(tf.keras.layers.Layer):
    """ Seq2Seq 모델의 Decoder layer

    Translation에서는 시작 토큰(e.g., <SOS>)과 인코더의 벡터를 받아 단어를 예측하고, 
    예측한 단어를 다시 입력으로 사용해 연속적으로 예측함. 

    시계열 예측에서는 t시점의 데이터를 시작 토큰으로 주고 이후 n-step을 예측함

    인코더와 동일하게 GRU를 사용    

    Attention을 사용할 경우 디코더의 state를 계산할 때 이전 state와 input만을 참고하는 것이 아니라 
    인코더의 모든 step에서의 출력을 추가로 참고하여 state를 계산함.

    다양한 종류의 Attention이 있으며 여기서는 dot-product attention을 사용

    Attributes:
        units: (int) GRU의 차원
        dropout: (float) dropout
        attention: (bool) Attenion layer를 사용할 지 여부
    """
    def __init__(self, units, dropout, attention=False, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)
        self.units = units
        self.attention = attention

        self.gru = tf.keras.layers.GRU(units, 
                                       return_sequences=True,
                                       return_state=True, dropout=dropout)
        if attention:
            self.attn = tf.keras.layers.Attention()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, hidden, enc_output, training=False):
        # inputs => (batch, label_width, feature_num)
        # hidden => (batch, enc_dim)
        # enc_output => (batch, input_width, enc_dim)

        outputs, state = self.gru(inputs, initial_state=hidden)
        if self.attn:
            # Luong attention
            # context_vec => (batch, label_width, dec_dim)
            context_vec = self.attn([outputs, enc_output])
            # outputs => (batch, label_width, dec_dim*2)
            outputs = tf.concat([outputs, context_vec], axis=-1)
        outputs = self.dense(outputs)
        return tf.squeeze(outputs, axis=-1)

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({"class_name": "Decoder", "config": {"units": self.units}})
        return config


def get_model(units, input_width, input_feature,
              label_width, label_feature, dropout, attention=False):
    tf.keras.backend.clear_session()
    enc_input = tf.keras.Input(shape=(input_width, input_feature))
    encoder = Encoder(units, dropout)
    dec_input = tf.keras.Input(shape=(None, label_feature))
    decoder = Decoder(units, dropout, attention=attention)

    enc_outputs, enc_state = encoder(enc_input)
    dec_outputs = decoder(dec_input, enc_state, enc_outputs)

    model = tf.keras.Model((enc_input, dec_input), dec_outputs)
    model.compile(loss='mse', optimizer='adam')
    return model