
import sys
import os


import numpy as np
from scipy.misc import imresize
from scipy.stats import entropy
from scipy.spatial.distance import directed_hausdorff, euclidean
import scipy.io as sio

from fastdtw import fastdtw
import editdistance
import requests, zipfile, io



try:
	import matlab
	import matlab.engine
	from io import StringIO
except ImportError:
	print("some function won't work without matlab & matlab API installed")


def make_engine():
	"""
			works only if you have matlab & matlab API installed

				cd MATLAB_ROOT/extern/engines/python/
				sudo python setup.py install
	"""

	try:
		eng = matlab.engine.start_matlab()

		# downloading ScanMatch if it doesn't exist
		sm_path = 'matlab/ScanMatch/'
		mm_path = 'matlab/MultiMatchToolbox/'
		if not os.path.isdir(sm_path):
			os.makedirs(sm_path)
			url = 'https://seis.bristol.ac.uk/~psidg/ScanMatch/ScanMatch.zip'
			print("downloading ScanMatch from the authers' website")
			r = requests.get(url, stream=True)
			z = zipfile.ZipFile(io.BytesIO(r.content))
			z.extractall(sm_path)

		if not os.path.isdir(mm_path):
			os.makedirs(mm_path)
			url = 'http://dev.humlab.lu.se/www-transfer/people/marcus-nystrom/MultiMatchToolbox.zip'
			print("downloading MultiMatchToolBox from the authers' website")
			r = requests.get(url, stream=True)
			z = zipfile.ZipFile(io.BytesIO(r.content))
			z.extractall('matlab/')

		eng.addpath(mm_path)
		eng.cd(sm_path)

		return eng
	except Exception as e:
		print(e)
		return


"""
Collection of common saliency metrics

If you're using this code, please don't forget to cite the original code
as mentioned in each function doc.

"""

def NSS(saliency_map, fixation_map):
	""""
	normalized scanpath saliency between two different
	saliency maps as the mean value of the normalized saliency map at
	fixation locations.

		Computer NSS score.
		:param saliency_map : predicted saliency map
		:param fixation_map : ground truth saliency map.
		:return score: float : score

	"""
	if isinstance(saliency_map, np.ndarray):
		saliency_map = np.array(saliency_map)
	if isinstance(fixation_map, np.ndarray):
		fixation_map = np.array(fixation_map)

	if saliency_map.size != fixation_map.size:
		saliency_map = imresize(saliency_map, fixation_map.shape)


	MAP = (saliency_map - saliency_map.mean()) / (saliency_map.std())
	mask = fixation_map.astype(np.bool)

	score =  MAP[mask].mean()

	return score


def CC(saliency_map, saliency_map_gt):
	"""
	This finds the linear correlation coefficient between two different
	saliency maps (also called Pearson's linear coefficient).
	score=1 or -1 means the maps are correlated
	score=0 means the maps are completely uncorrelated

	saliencyMap1 and saliencyMap2 are 2 real-valued matrices

		Computer CC score .
		:param saliency_map : first saliency map
		:param saliency_map_gt : second  saliency map.
		:return score: float : score

	"""
	if isinstance(saliency_map, np.ndarray):
		saliency_map = np.array(saliency_map, dtype=np.float32)
	elif saliency_map.dtype != np.float32:
		saliency_map = saliency_map.astype(np.float32)

	if isinstance(saliency_map_gt, np.ndarray):
		saliency_map_gt = np.array(saliency_map_gt, dtype=np.float32)
	elif saliency_map.dtype != np.float32:
		saliency_map_gt = saliency_map_gt.astype(np.float32)

	if saliency_map.size != saliency_map_gt.size:
		saliency_map = imresize(saliency_map, saliency_map_gt.shape)

	saliency_map = (saliency_map - saliency_map.mean()) / (saliency_map.std())
	saliency_map_gt = (saliency_map_gt - saliency_map_gt.mean()) / (saliency_map_gt.std())

	score = np.corrcoef(saliency_map.flatten(),saliency_map_gt.flatten())[0][1]

	return score


def EMD():
	"""

	if you are using this function, please cite the following papers:

		http://www.ariel.ac.il/sites/ofirpele/fastemd/code/
		https://github.com/wmayner/pyemd

	"""
	pass




def KLdiv(saliency_map, saliency_map_gt):
	"""
	This finds the KL-divergence between two different saliency maps when
	viewed as distributions: it is a non-symmetric measure of the information
	lost when saliencyMap is used to estimate fixationMap.

		Computer KL-divergence.
		:param saliency_map : predicted saliency map
		:param fixation_map : ground truth saliency map.
		:return score: float : score

	"""

	if isinstance(saliency_map, np.ndarray):
		saliency_map = np.array(saliency_map, dtype=np.float32)
	elif saliency_map.dtype != np.float32:
		saliency_map = saliency_map.astype(np.float32)

	if isinstance(saliency_map_gt, np.ndarray):
		saliency_map_gt = np.array(saliency_map_gt, dtype=np.float32)
	elif fixation_map.dtype != np.float32:
		saliency_map_gt = saliency_map_gt.astype(np.float32)

	# the function will normalize maps before computing Kld
	score = entropy(saliency_map.flatten(), saliency_map_gt.flatten())
	return score



def AUC(saliency_map, fixation_map):
	"""Computes AUC for given saliency map 'saliency_map' and given
	fixation map 'fixation_map'
	"""
	def area_under_curve(predicted, actual, labelset):
		def roc_curve(predicted, actual, cls):
			si = np.argsort(-predicted)
			tp = np.cumsum(np.single(actual[si]==cls))
			fp = np.cumsum(np.single(actual[si]!=cls))
			tp = tp/np.sum(actual==cls)
			fp = fp/np.sum(actual!=cls)
			tp = np.hstack((0.0, tp, 1.0))
			fp = np.hstack((0.0, fp, 1.0))
			return tp, fp
		def auc_from_roc(tp, fp):
			h = np.diff(fp)
			auc = np.sum(h*(tp[1:]+tp[:-1]))/2.0
			return auc

		tp, fp = roc_curve(predicted, actual, np.max(labelset))
		auc = auc_from_roc(tp, fp)
		return auc

	fixation_map = (fixation_map>0.7).astype(int)
	salShape = saliency_map.shape
	fixShape = fixation_map.shape

	predicted = saliency_map.reshape(salShape[0]*salShape[1], -1, order='F').flatten()
	actual = fixation_map.reshape(fixShape[0]*fixShape[1], -1, order='F').flatten()
	labelset = np.arange(2)

	return area_under_curve(predicted, actual, labelset)



def SAUC(saliency_map, fixation_map, shuf_map=np.zeros((480,640)), step_size=.01):
	"""
		please cite:  https://github.com/NUS-VIP/salicon-evaluation

		calculates shuffled-AUC score.

		:param salinecy_map : predicted saliency map
		:param fixation_map : ground truth saliency map.
		:return score: int : score

	"""

	saliency_map -= np.min(saliency_map)
	fixation_map = np.vstack(np.where(fixation_map!=0)).T

	if np.max(saliency_map) > 0:
		saliency_map = saliency_map / np.max(saliency_map)
	Sth = np.asarray([ saliency_map[y-1][x-1] for y,x in fixation_map ])
	Nfixations = len(fixation_map)

	others = np.copy(shuf_map)
	for y,x in fixation_map:
		others[y-1][x-1] = 0

	ind = np.nonzero(others) # find fixation locations on other images
	nFix = shuf_map[ind]
	randfix = saliency_map[ind]
	Nothers = sum(nFix)

	allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),step_size)
	allthreshes = allthreshes[::-1]
	tp = np.zeros(len(allthreshes)+2)
	fp = np.zeros(len(allthreshes)+2)
	tp[-1]=1.0
	fp[-1]=1.0
	tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
	fp[1:-1]=[float(np.sum(nFix[randfix >= thresh]))/Nothers for thresh in allthreshes]

	score = np.trapz(tp,fp)
	return score


def euclidean_distance(P,Q, **kwargs):
	if isinstance(P, np.ndarray):
		P = np.array(P, dtype=np.float32)
	elif P.dtype != np.float32:
		P = P.astype(np.float32)

	if isinstance(Q, np.ndarray):
		Q = np.array(Q, dtype=np.float32)
	elif Q.dtype != np.float32:
		Q = Q.astype(np.float32)

	if P.shape == Q.shape:
		return np.sqrt(np.sum((P-Q)**2))
	return False



def hausdorff_distance(P, Q, **kwargs):
	if isinstance(P, np.ndarray):
		P = np.array(P, dtype=np.float32)
	elif P.dtype != np.float32:
		P = P.astype(np.float32)

	if isinstance(Q, np.ndarray):
		Q = np.array(Q, dtype=np.float32)
	elif Q.dtype != np.float32:
		Q = Q.astype(np.float32)

	return max(directed_hausdorff(P, Q)[0], directed_hausdorff(Q, P)[0])

def frechet_distance(P, Q, **kwargs):
	""" Computes the discrete frechet distance between two polygonal lines
	Algorithm: http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
	P and Q are arrays of 2-element arrays (points)
	"""
	if isinstance(P, np.ndarray):
		P = np.array(P, dtype=np.float32)
	elif P.dtype != np.float32:
		P = P.astype(np.float32)

	if isinstance(Q, np.ndarray):
		Q = np.array(Q, dtype=np.float32)
	elif Q.dtype != np.float32:
		Q = Q.astype(np.float32)

	def euc_dist(pt1,pt2):
		return np.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0])+(pt2[1]-pt1[1])*(pt2[1]-pt1[1]))

	def _c(ca,i,j,P,Q):
		if ca[i,j] > -1:
			return ca[i,j]
		elif i == 0 and j == 0:
			ca[i,j] = euc_dist(P[0],Q[0])
		elif i > 0 and j == 0:
			ca[i,j] = max(_c(ca,i-1,0,P,Q),euc_dist(P[i],Q[0]))
		elif i == 0 and j > 0:
			ca[i,j] = max(_c(ca,0,j-1,P,Q),euc_dist(P[0],Q[j]))
		elif i > 0 and j > 0:
			ca[i,j] = max(
						min(_c(ca,i-1,j,P,Q), _c(ca,i-1,j-1,P,Q), _c(ca,i,j-1,P,Q)),
							euc_dist(P[i],Q[j]))
		else:
			ca[i,j] = float("inf")
		return ca[i,j]
	ca = np.ones((len(P),len(Q)))
	ca = np.multiply(ca,-1)
	return _c(ca,len(P)-1,len(Q)-1,P,Q)



def levenshtein_distance(P,Q, height, width, Xbins=12, Ybins = 8, **kwargs):
	"""
		Levenshtein distance
	"""
	def _scanpath_to_string(scanpath, height, width, Xbins, Ybins):
		"""
				a b c d ...
			A
			B
			C
			D

			returns Aa
		"""
		height_step, width_step = height//Ybins, width//Xbins
		string = ''
		for i in range(scanpath.shape[0]):
			fixation = scanpath[i].astype(np.int32)
			corrs_x = chr(65 + fixation[1]//height_step)
			corrs_y = chr(97 + fixation[0]//width_step)
			string += (corrs_x + corrs_y)
		return string

	str_1 = _scanpath_to_string(P, height, width, Xbins, Ybins)
	str_2 = _scanpath_to_string(Q, height, width, Xbins, Ybins)

	return editdistance.eval(str_1, str_2)



def DTW(P, Q, **kwargs):
	dist, _ =  fastdtw(P, Q, dist=euclidean)
	return dist


def time_delay_embedding_distance(
		P,
		Q,

		# options
		k=3,  # time-embedding vector dimension
		distance_mode='Mean', **kwargs
		):

	"""
		code reference:
			https://github.com/dariozanca/FixaTons/
			https://arxiv.org/abs/1802.02534

		metric: Simulating Human Saccadic Scanpaths on Natural Images.
				 wei wang etal.
	"""

	# P and Q can have different lenghts
	# They are list of fixations, that is couple of coordinates
	# k must be shorter than both lists lenghts

	# we check for k be smaller or equal then the lenghts of the two input scanpaths
	if len(P) < k or len(Q) < k:
		print('ERROR: Too large value for the time-embedding vector dimension')
		return False

	# create time-embedding vectors for both scanpaths

	P_vectors = []
	for i in np.arange(0, len(P) - k + 1):
		P_vectors.append(P[i:i + k])

	Q_vectors = []
	for i in np.arange(0, len(Q) - k + 1):
		Q_vectors.append(Q[i:i + k])

	# in the following cicles, for each k-vector from the simulated scanpath
	# we look for the k-vector from humans, the one of minumum distance
	# and we save the value of such a distance, divided by k

	distances = []

	for s_k_vec in Q_vectors:

		# find human k-vec of minimum distance

		norms = []

		for h_k_vec in P_vectors:
			d = np.linalg.norm(euclidean_distance(s_k_vec, h_k_vec))
			norms.append(d)

		distances.append(min(norms) / k)

	# at this point, the list "distances" contains the value of
	# minumum distance for each simulated k-vec
	# according to the distance_mode, here we compute the similarity
	# between the two scanpaths.

	if distance_mode == 'Mean':
		return sum(distances) / len(distances)
	elif distance_mode == 'Hausdorff':
		return max(distances)
	else:
		print('ERROR: distance mode not defined.')
		return False




def MultiMatch(matlab_engine, P, Q, height, width, check=False, **kwargs):
	"""
		works only if you have matlab & matlab API installed

		1 )
			cd MATLAB_ROOT/extern/engines/python/
			sudo python setup.py install

	"""
	try:
		if 'matlab' not in sys.modules:
			print('This function requires MATLAB API installed.\
					cd MATLAB_ROOT/extern/engines/python/ \
					sudo python setup.py install')
			return

		if P.shape[1] == 2:
			P = np.hstack([P,  np.random.rand(P.shape[0],1)])
			Q = np.hstack([Q,  np.random.rand(Q.shape[0],1)])

		P = matlab.double(P.tolist())
		Q = matlab.double(Q.tolist())
		# if (check) and ('metrics/MultiMatchToolbox' not in eng.pwd()):
		# 	eng.cd('metrics/MultiMatchToolbox/')
		# print(P,Q)

		size = matlab.double([width, height])

		return matlab_engine.doComparison(P,Q, size, stdout=StringIO())
		# return np.array(result).squeeze()
	except Exception as e:
		print(e)
		return [np.nan, np.nan, np.nan, np.nan, np.nan]

def ScanMatch(matlab_engine, P,Q, height, width, Xbins=12, Ybins = 8,
				ScanMatchInfo='ScanMatchInfo_OSIE.mat', **kwargs):
	"""
		ScanMatch
		You need to creat ScanMatchInfo file before hand in the matlab yourself.

		for more information have look at:
			https://seis.bristol.ac.uk/~psidg/ScanMatch/

	"""


	def _load_scanmatch_info(matlab_engine, ScanMatchInfo):
		try:
			matlab_engine.load(ScanMatchInfo, nargout=0)
			return True

		except Exception as e:
			print(e)
			return False

	def _scanpath_to_string(scanpath, height, width, Xbins, Ybins):
		"""
				a b c d ...
			A
			B
			C
			D

			returns Aa
		"""
		height_step, width_step = height//Ybins, width//Xbins
		string = ''
		for i in range(scanpath.shape[0]):
			fixation = scanpath[i].astype(np.int32)
			corrs_x = chr(97 + fixation[1]//height_step)
			corrs_y = chr(65 + fixation[0]//width_step)
			string += (corrs_x + corrs_y)
		return string



	try:
		if 'matlab' not in sys.modules:
			print('This function requires MATLAB API installed.\
					cd MATLAB_ROOT/extern/engines/python/ \
					sudo python setup.py install')
			return

		P = _scanpath_to_string(P, height, width, Xbins, Ybins)
		Q = _scanpath_to_string(Q, height, width, Xbins, Ybins)

		# loading variables in matlab
		matlab_engine.workspace['seq1'] = P
		matlab_engine.workspace['seq2'] = Q

		if 'ScanMatchInfo' not in matlab_engine.eval('who'):
			if not _load_scanmatch_info(matlab_engine, ScanMatchInfo):
				print("The MAT File doesn't exist")
				return False


		return matlab_engine.eval('ScanMatch(seq1, seq2, ScanMatchInfo)', stdout=StringIO())
		# return matlab_engine.ScanMatch(P, Q, stdout=StringIO())
		# return np.array(result).squeeze()
	except Exception as e:
		print(e)
		return np.nan


