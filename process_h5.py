import re
from dataclasses import dataclass
import typing
from pathlib import Path
import concurrent.futures as cf
from functools import partial

import h5py
import numpy as np
import pandas as pd


def process(h5f):
	model = []
	with h5py.File(h5f, 'r') as file:
		for key in file.keys():
			if 'de' in key:
				layer = [file.get(f'{key}/{key}').get(x) for x in file.get(f'{key}/{key}').keys()]
				model.append(np.vstack(tuple(layer)))
	return np.vstack(tuple(model)).flatten()


def _threading_key_extractor(fi, r):
	return r.search(fi.parts[1]).group()


# TODO refactor _threading_key_extractor and _threading_dict_gen into single function/call
# dict.fromkeys uses single instance of `default value` for ALL keys, so if given [] there is one shared [] for ALL keys
def _threading_dict_gen(fi, fd, f):
	fd.get(f(fi)).append(fi)


@dataclass
class ModelData:
	"""Class for keeping track of model data"""
	name: str
	n_neurons: int
	dropout_rate: float
	opt: str
	lr: 'typing.Any'
	stats_file: str
	h5_files: typing.List[str]
	weights: 'typing.Any' = None
	stats: 'typing.Any' = None

	def populate_weights(self):
		# TODO sort self.h5_files *before* creating array
		self.weights = np.array([process(h5) for h5 in self.h5_files])

	def populate_stats(self):
		self.stats = pd.read_csv(self.stats_file)

	def get_hyp(self):
		return {'op': self.opt, 'nu': self.n_neurons, 'do': self.dropout_rate, 'lr': self.lr}


def get_hparams_from_name(file_name: str):
	optimizer_key = ['Adam', 'SGD', 'RMSprop']
	neurons, dropout, opt, lr = file_name.split('_')
	dropout = np.float32(dropout.replace('-', '.'))
	lr = np.float32(lr.replace('-', '.'))
	return int(neurons), dropout, optimizer_key[int(opt)], lr


def gen_model_data_list(fd):
	csv_dir = Path('csv/')
	csv_files = csv_dir.glob('*')
	model_dat = []
	for fi in csv_files:
		if fi.is_dir():
			pass
		m_name = fi.parts[-1]
		try:
			m = ModelData(m_name, *get_hparams_from_name(m_name), fi, fd.get(m_name), None, None)
			model_dat.append(m)
		except:
			print(fi)

	return model_dat


def run(directory):
	"""directory = Path('checkpoints/')"""
	directory = Path(directory)
	h5_key_re = re.compile(r'((?<=model_[0-9]_)|(?<=model_[0-9][0-9]_)).*(?=\.h5)')
	h5_file_list = list(directory.glob('*.h5'))
	_threading_key_extractor_partial = partial(_threading_key_extractor, r=h5_key_re)

	with cf.ThreadPoolExecutor() as executor:
		p = executor.map(_threading_key_extractor_partial,  h5_file_list)

	file_dict = {k: [] for k in set(p)}
	_threading_dict_gen_partial = partial(_threading_dict_gen, fd=file_dict, f=_threading_key_extractor_partial)

	with cf.ThreadPoolExecutor() as executor:
		p = executor.map(_threading_dict_gen_partial, h5_file_list)

	return gen_model_data_list(file_dict)
