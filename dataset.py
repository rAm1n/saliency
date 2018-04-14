


import json
from scipy.misc import imread
from PIL import Image
import numpy as np
import zipfile
import tarfile
import os
import requests



CONFIG = {
	'data_path' : os.path.expanduser('~/tmp/saliency/'),
	'dataset_json' : os.path.join(os.path.dirname(__file__), 'data/dataset.json'),
	'auto_download' : True,
}


class SaliencyDataset():
	def __init__(self, name, config=CONFIG):
		self.name =  name.upper()
		self.config = config
		self.sequence = None
		self._download_or_load()

	def __repr__(self):
		return 'Dataset object - {0}'.format(self.name)

	def __str__(self):
		return 'Dataset object - {0}'.format(self.name)

	def __len__(self):
		return len(self.data)

	def _download_or_load(self):


		try:
			dataset_file = self.config['dataset_json']
			with open(dataset_file, 'r') as f_handle:
				data = json.load(f_handle)[self.name]
			for key, value in data.items():
				setattr(SaliencyDataset, key, value)

		except KeyError:
			print('{0} has not been converted yet'.format(self.name))

		except Exception as x:
			print(x)
			print('something went wrong')
			exit()

		try:
			self.directory = os.path.join(self.config['data_path'], self.name)
			if not os.path.isdir(self.directory):
					os.makedirs(self.directory)
		except OSError as e:
				raise e

		self._load('data')

	def _download(self, url, path, extract=False):
		try:
			print('downloading - {0}'.format(url))
			def save_response_content(response, destination):
				CHUNK_SIZE = 32768
				try:
					with open(destination, "wb") as f:
						for chunk in response.iter_content(CHUNK_SIZE):
							if chunk: # filter out keep-alive new chunks
								f.write(chunk)
				except Exception as x:
					print(x)
			if ("drive.google.com" in url):
				def get_confirm_token(response):
					for key, value in response.cookies.items():
						if key.startswith('download_warning'):
							return value
					return None

				filename = url.split('=')[-1] + '.zip'
				file_extension = 'zip'
				destination = os.path.join(self.config['data_path'], self.name, filename)

				session = requests.Session()
				response = session.get(url, stream = True)
				token = get_confirm_token(response)

				if token:
					params = { 'confirm' : token }
					response = session.get(url, params = params, stream = True)
			else:
				headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
				session = requests.Session()
				response = session.get(url, stream = True, headers=headers)

				filename = url.split('/')[-1]
				file_extension = filename.split('.')[-1]
				destination = os.path.join(path, filename)

			save_response_content(response, destination)

			if file_extension == 'zip':
				zip_ref = zipfile.ZipFile(destination, 'r')
				zip_ref.extractall(path)
				zip_ref.close()
				os.remove(destination)
			elif file_extension == 'tgz':
				tar = tarfile.open(destination, 'r')
				tar.extractall(path)
				tar.close()
				os.remove(destination)

		except Exception as x:
			print(x)
			os.rmdir(path)


	def _load(self, key):
		try:
			if key not in self.url:
				key = 'data'

			sub_dir = os.path.join(self.directory, key)
			if not os.path.isdir(sub_dir): # download
				try:
					os.makedirs(sub_dir)
					self._download(self.url[key], sub_dir)
				except Exception as x:
					print(x)

			if (key == 'sequence') and ( self.sequence is None) :
				npz_file = os.path.join(sub_dir, '{0}.npz'.format(self.name))
				with open(npz_file, 'rb') as f_handle:
					self.sequence = np.load(f_handle)

		except Exception as x:
			print(x)

	def get(self, data_type, **kargs):
		result = list()
		for idx, img in enumerate(self.data):
			if data_type=='sequence':
				self._load('sequence')
				tmp = list()
				for user in self.sequence[idx]:
					user = np.array(user)
					if 'percentile' in kargs:
						if kargs['percentile']:
							if(user.shape)[0] == 0:
								continue
							_sample = user[:,:2] / self.img_size
							user = np.concatenate((_sample, user[:,2:]), axis=1)
					if 'modify' in kargs:
						if kargs['modify']== 'fix' :
							if 'percentile' in kargs:
								if kargs['percentile']:
									mask_greater = _sample > 1.0
									mask_smaller = _sample < 0.0
									_sample[mask_greater] = (1.0 - np.finfo(float).eps)
									_sample[mask_smaller] = np.finfo(float).eps
									user = np.concatenate((_sample, user[:,2:]), axis=1)
								else:
									# TODO
									print('fix was ignored, only works in percentile mode.')
							else:
									# TODO
								print('fix was ignored, only works in percentile mode.')
						elif kargs['modify'] == 'remove':
							if 'percentile' in kargs:
								if kargs['percentile']:
									user = user[user[:,0]<=(1 - np.finfo(float).eps), :]
									user = user[user[:,0]>=(np.finfo(float).eps), :]
									user = user[user[:,1]<=(1-np.finfo(float).eps), :]
									user = user[user[:,1]>=(np.finfo(float).eps), :]
								else:
									# TODO
									print('fix was ignored, only works in percentile mode.')
							else:
								# TODO
								print('fix was ignored, only works in percentile mode.')
					tmp.append(user)
				tmp = np.array(tmp)

			elif data_type =='heatmap':
				self._load('heatmap')
				path = os.path.join(self.directory, img['heatmap'])
				if os.path.isfile(path):
					tmp = imread(path)

			elif data_type == 'heatmap_path':
				self._load('heatmap')
				tmp = os.path.join(self.directory, img['heatmap'])

			elif data_type =='stimuli':
				path = os.path.join(self.directory, img['stimuli'])
				if os.path.isfile(path):
					tmp = imread(path)
					if tmp.ndim != 3:
						shape = tmp.shape
						tmp = np.array(Image.fromarray(tmp).convert('RGB').getdata()).reshape(shape + (3,))
			elif data_type == 'stimuli_path':
				tmp = os.path.join(self.directory, img['stimuli'])
			else:
				try:
					tmp = self.data[data_type]
				except Exception as x:
					return False
			result.append(tmp)

		result = np.asarray(result)
		return result
