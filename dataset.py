


import json
import pickle
import wget
from scipy.misc import imread
import numpy as np
import zipfile
import tarfile
import os
from tqdm import tqdm
import requests




CONFIG = {
	'data_path' : os.path.expanduser('~/tmp/saliency/'),
	'auto_download' : True,
	'json_directory' : 'http://saliency.raminfahimi.ir/json/'
}


# TODO add len in json and fix code


class SaliencyDataset():
	def __init__(self, name, config=CONFIG):
		self.name =  name
		self.config = config
		self._download_or_load()

	def __repr__(self):
		return 'Dataset object - {0}'.format(self.name)

	def __str__(self):
		return 'Dataset object - {0}'.format(self.name)

	def __len__(self):
		return len(self.data)

	def _download_or_load(self):

		try:
			directory = os.path.join(self.config['data_path'], self.name)
			self.directory = directory
			if os.path.isdir(directory):
				return self._load()
			os.makedirs(directory)
		except OSError as e:
				raise e

		# Starting Download
		# for url in URLs[self.name]:
		# 	self._download(url, unzip=True)
		pkl_url  = self.config['json_directory'] + '{0}.pkl'.format(self.name.upper()) 

		self._download(pkl_url)



		pkl_directory = os.path.join(self.config['data_path'], self.name)
		try:
			path = os.path.join(pkl_directory, '{0}.pkl'.format(self.name))
			f = open(path, 'rb')
			data = pickle.load(f)#, encoding='latin1')
			for url in data['url']:
				self._download(url, extract=True)
			f.close()
		except Exception as x:
			print(x)

		#loading data
		self._load()


	def _download(self, url, extract=False):
		try:
			print('downloading - {0}'.format(url))
			def save_response_content(response, destination):
				CHUNK_SIZE = 32768
				with open(destination, "wb") as f:
					for chunk in response.iter_content(CHUNK_SIZE):
						if chunk: # filter out keep-alive new chunks
							f.write(chunk)

			if ("drive.google.com" in url):
				def get_confirm_token(response):
					for key, value in response.cookies.items():
						if key.startswith('download_warning'):
							return value
					return None

				filename = url.split('=')[-1] + '.zip'
				destination = os.path.join(self.config['data_path'], self.name, filename)

				session = requests.Session()
				response = session.get(url, stream = True)
				token = get_confirm_token(response)

				if token:
					params = { 'confirm' : token }
					response = session.get(url, params = params, stream = True)    

			else:
				filename = url.split('/')[-1]
				destination = os.path.join(self.config['data_path'], self.name, filename)

				session = requests.Session()
				response = session.get(url, stream = True)
				# wget.download(url, destination)

			save_response_content(response, destination)

			if extract:
				directory = os.path.dirname(destination)
				_ , file_extension = os.path.splitext(destination)
				print(destination, file_extension, directory)
				if file_extension == '.zip':
					zip_ref = zipfile.ZipFile(destination, 'r')
					zip_ref.extractall(directory)
					zip_ref.close()
				else:
					tar = tarfile.open(destination, 'r')
					tar.extractall(directory)
					tar.close()
				os.remove(destination)

		except Exception as x:
			print(x)
			directory = os.path.dirname(destination)
			os.rmdir(directory)


	def _load(self):
		pkl_directory = os.path.join(self.config['data_path'], self.name)
		try:
			path = os.path.join(pkl_directory, '{0}.pkl'.format(self.name))
			f = open(path, 'rb')
			data = pickle.load(f)#, encoding='latin1')
			for key,value in data.items():
				setattr(SaliencyDataset, key, value)
			# pre-processing data
			self.len = len(data)
		except Exception as x:
			print(x)

	def get(self, data_type, **kargs):
		result = list()
		for img in tqdm(self.data):
			if data_type=='sequence':
				tmp = list()
				for user in img['sequence']:
					user = np.array(user)
					if 'percentile' in kargs:
						if kargs['percentile']:
							if(user.shape)[0] == 0:
								continue
							_sample = user[:,:2] / self.size
							user = np.concatenate((_sample, user[:,2:]), axis=1)
					if 'modify' in kargs:
						if kargs['modify']== 'fix' :
							if 'percentile' in kargs:
								if kargs['percentile']:
									mask_greater = _sample > 1.0
									mask_smaller = _sample < 0.0
									_sample[mask_greater] = 0.999999
									_sample[mask_smaller] = 0.000001
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
									user = user[user[:,0]<=0.99999, :]
									user = user[user[:,0]>=0.00001, :]
									user = user[user[:,1]<=0.99999, :]
									user = user[user[:,1]>=0.00001, :]
								else:
									# TODO
									print('fix was ignored, only works in percentile mode.')
							else:
								# TODO
								print('fix was ignored, only works in percentile mode.')
					tmp.append(user)
				tmp = np.array(tmp)
					# else:
					# 	tmp = np.array([np.array(user) for user in img['sequence']])

			elif data_type =='heatmap':
				path = os.path.join(self.directory, img['heatmap'])
				print path
				if os.path.isfile(path):
					tmp = imread(path)
				else:
					tmp = np.fromstring( img['heatmap'].decode('base64'), \
						dtype='int8').reshape(self.size)
	
			elif data_type == 'heatmap_path':
							tmp = os.path.join(self.directory, img['heatmap'])

			elif data_type =='stimuli':
				path = os.path.join(self.directory, img['stimuli'])
				if os.path.isfile(path):
					tmp = imread(path)
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



