import tensorflow as tf


class DataSlicer():
    """ 시계열 데이터를 학습에 적합하도록 슬라이싱 해주는 클래스
    
    동일한 길이의 Input data와 Label data를 받아 width에 맞춰 슬라이싱

    Attributes:
        input_data: 입력으로 사용될 데이터
        label_data: 라벨로 사용될 데이터
        input_width: 원하는 입력 데이터의 time step
        label_width: 원하는 라벨 데이터의 time step
        train_split: 학습 데이터 비율
        val_split: 검증 데이터 비율
        test_split: 테스트 데이터 비율

    Examples:
        >>> ds = DataSlicer(np.arange(20), np.arange(20), 5, 2)
        >>> list(ds.train.take(2))
        [(<tf.Tensor: shape=(5,), dtype=int64, numpy=array([0, 1, 2, 3, 4])>,
          <tf.Tensor: shape=(2,), dtype=int64, numpy=array([5, 6])>),
         (<tf.Tensor: shape=(5,), dtype=int64, numpy=array([1, 2, 3, 4, 5])>,
          <tf.Tensor: shape=(2,), dtype=int64, numpy=array([6, 7])>)]
    """

    def __init__(self, input_data, label_data, input_width, label_width,
                 teacher_force=False,
                 train_split=0.6, val_split=0.2, test_split=0.2):
        self.input_data = input_data
        self.label_data = label_data
        self.input_width = input_width
        self.label_width = label_width
        self.teacher_force = teacher_force
        self.training = label_data is not None and label_width != 0
        self.window_size = input_width + label_width
        if train_split+val_split+test_split!=1.0:
            raise ValueError(f"Split values error, sum of split values must be 1. train, val, test split values {train_split}, {val_split}, {test_split}.")
        self.data_size = len(input_data)
        self.train_size = int(len(input_data)*train_split)
        self.val_size = int(len(input_data)*val_split)
        self.test_size = len(input_data)-self.train_size-self.val_size


    def _make_dataset(self, input_data, label_data):
        def sub_to_branch(sub):
            return sub.batch(self.window_size, drop_remainder=True)
        
        def slice_input_label(input_batch, label_batch):
            if self.teacher_force:
                return (input_batch[:self.input_width], label_batch[-self.label_width-1:-1]), label_batch[-self.label_width:]
            else:
                return input_batch[:self.input_width], label_batch[-self.label_width:]

        input_ds = tf.data.Dataset.from_tensor_slices(input_data)
        input_ds = input_ds.window(self.window_size, shift=1).flat_map(sub_to_branch)

        if self.training:
            label_ds = tf.data.Dataset.from_tensor_slices(label_data)
            label_ds = label_ds.window(self.window_size, shift=1).flat_map(sub_to_branch)
            dataset = tf.data.Dataset.zip((input_ds, label_ds)).map(slice_input_label)
        else:
            dataset = input_ds

        return dataset

    @property
    def train(self):
        return self._make_dataset(self.input_data[:self.train_size], self.label_data[:self.train_size])
    
    @property
    def val(self):
        return self._make_dataset(self.input_data[self.train_size:self.train_size+self.val_size],
                                  self.label_data[self.train_size:self.train_size+self.val_size])
                        
    @property
    def test(self):
        return self._make_dataset(self.input_data[-self.test_size:], self.label_data[-self.test_size:])

    @property
    def total(self):
        return self._make_dataset(self.input_data, self.label_data)
