from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

class DataGenerator():
    ''' 데이터 생성기 
    시계열 형태의 csv 데이터를 window 크기에 맞춰 input과 label로 분리
    예:
        a   b   c
    0   1   10  100
    1   2   20  200
    2   3   30  300
    3   4   40  400
    ...
    29  30  300 3000

    위와 같은 형태의 csv데이터를 raw_data,
    feature_cols=['a', 'b'], label_cols='c'이고,

    input_width=5, label_width=3, shift=1, stride=1 이면,
    [([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]], [600, 700, 800]),
        ([[2, 20], [3, 30], [4, 40], [5, 50], [6, 60]], [700, 800, 900]),
        ...
        ([[23, 230], [24, 240], [25, 250], [26, 260], [27, 270]], [2800, 2900, 3000])]

        형태의 tf.data.Dataset를 만든다.

        split은 전체 데이터를 train/val/test로 나눌 비율을 정한다. (합은 1)
    '''

    def __init__(self, raw_data, input_width, label_width, 
                feature_cols, label_cols, shift=1, stride=1, norm=True,
                train_split=0.75, val_split=0.15, test_split=0.2):
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.stride = stride
        self.norm = norm
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.raw_data = raw_data
        self.window_size = input_width+label_width
        self.data_size = len(raw_data)-self.window_size+1
        self.train_size = int(train_split*self.data_size)
        self.val_size = int(val_split*self.data_size)
        self.test_size = self.data_size-self.train_size-self.val_size
        self.train_pos = self.train_size
        self.val_pos = self.train_pos+self.val_size
        self.test_pos = self.val_pos+self.test_size

        self.make_dataset()
    
    def __repr__(self):
        return '\n'.join([f'total window: {[_*self.stride for _ in range(self.window_size)]}',
                        f'input width: {[_*self.stride for _ in range(self.input_width)]}',
                        f'label width: {[_*self.stride for _ in range(self.input_width, self.window_size)]}'])

    def make_dataset(self):
        # dataset.window를 사용하면 sub dataset을 반환한다.
        # flat map을 사용해 tensor를 바로 받을 수 있도록 하는 함수
        def sub_to_branch(sub):
            return sub.batch(self.window_size, drop_remainder=True)
        
        # Decoder의 첫 번째 입력은 일반적으로 시작 신호를 주는데,
        # 여기서는 첫 label의 한 단계 이전 값을 사용함
        # 예를 들어, label이 [4, 5, 6]이면 decoder input은 [3, 4, 5]이고
        # 학습 시에는 교사 강요(teacher forcing)를 사용하기 때문에 모두 입력으로 사용하고
        # 예측 시에는 [3]만 입력 신호로 사용하고 이후는 예측 값을 다시 입력으로 사용한다.
        def slice_feature_label(feature_batch, label_batch):
            return ((feature_batch[:self.input_width], label_batch[-self.label_width-1:-1]),
                        label_batch[-self.label_width:])
        if self.norm:
            scaler = MinMaxScaler()
            feature_data = scaler.fit_transform(self.raw_data[self.feature_cols].values)
            label_data = tf.squeeze(scaler.fit_transform(self.raw_data[[self.label_cols]].values))
        else:
            feature_data = self.raw_data[self.feature_cols].values
            label_data = self.raw_data[self.label_cols].values

        feature_ds = tf.data.Dataset.from_tensor_slices(feature_data)
        feature_ds = feature_ds.window(self.window_size, shift=self.shift, stride=self.stride).flat_map(sub_to_branch)
        label_ds = tf.data.Dataset.from_tensor_slices(label_data)
        label_ds = label_ds.window(self.window_size, shift=self.shift, stride=self.stride).flat_map(sub_to_branch)

        dataset = tf.data.Dataset.zip((feature_ds, label_ds)).map(slice_feature_label)
        self.dataset = dataset
        self.train = dataset.take(self.train_size)
        self.val = dataset.skip(self.train_size).take(self.val_size)
        self.test = dataset.skip(self.train_size).skip(self.val_size)
    