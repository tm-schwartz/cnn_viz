# cspell: disable
import tensorflow as tf
from tensorflow import keras
import numpy as np
import concurrent.futures


physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

mnist = tf.keras.datasets.mnist

# subset of data

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

indices = np.random.randint(0, len(x_train), size=10000)

x_train, y_train = x_train[indices], y_train[indices]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(17)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(17)

# call backs

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', mode='min', patience=5, min_delta=0.001)


class CheckpointCallback(tf.keras.callbacks.Callback):
	def __init__(self, fn):
		super(CheckpointCallback, self).__init__()
		self.fn = fn
		self.epoch = 0

	def on_epoch_end(self, batch, logs=None):
		self.model.save_weights(
		    f'checkpoints/10-07-20/model_{self.epoch}_{self.fn}.h5')
		self.epoch += 1


def get_m(hyp):
	if hyp.get('op') == 'Adam':
		opt = keras.optimizers.Adam
	elif hyp.get('op') == 'SGD':
		opt = keras.optimizers.SGD
	else:
		opt = keras.optimizers.RMSprop

	model = tf.keras.models.Sequential([
		tf.keras.layers.Flatten(dtype='float64'),
		tf.keras.layers.Dense(hyp['nu'], activation='relu', dtype='float64'),
		tf.keras.layers.Dropout(hyp['do'], dtype='float64'),
		tf.keras.layers.Dense(10, dtype='float64')
	])
	model.compile(
    optimizer=opt(
        learning_rate=hyp['lr']),
        loss=loss,
         metrics=['accuracy'])
	save_name = f"{int(hyp['nu'])}_{str(hyp['do']).replace('.', '-')}_{hyp['op_i']}_{str(hyp['lr']).replace('.', '-')}"
	try:
		model.fit(train_dataset, epochs=20,
                        validation_data=test_dataset, callbacks=[early_stopping,
                            CheckpointCallback(save_name)])
	except Exception as e:
		print(e)
# keras.callbacks.CSVLogger('csv/' + save_name, append=True)], verbose=1)

def gen_hparams():
	HP_NUM_UNITS= np.linspace(50, 200, 6)
	HP_DROPOUT= np.linspace(.1, .5, num=6)
	HP_OPTIMIZER= ['Adam', 'SGD', 'RMSprop']
	HP_LR= np.linspace(.001, .1, 6)
	for num_units in HP_NUM_UNITS:
		for dropout_rate in HP_DROPOUT:
			for learning in HP_LR:
				for i, optimizer in enumerate(HP_OPTIMIZER):
					yield {
						'nu': num_units,
						'do': dropout_rate,
						'op': optimizer,
						'lr': learning,
						'op_i': i
					}

def run():
	with concurrent.futures.ThreadPoolExecutor() as executor:
		p= executor.map(get_m, gen_hparams())
