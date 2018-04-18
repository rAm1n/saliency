
import sys


import numpy as np
from scipy.misc import imresize
from scipy.stats import entropy
from scipy.spatial.distance import directed_hausdorff, euclidean
from fastdtw import fastdtw
import editdistance


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
		eng.cd('metrics/MultiMatchToolbox/')
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
	mask = MAP.astype(bool)

	score =  MAP[mask].mean()

	return score


def CC(sal_map_1, sal_map_2):
	"""
	This finds the linear correlation coefficient between two different
	saliency maps (also called Pearson's linear coefficient).
	score=1 or -1 means the maps are correlated
	score=0 means the maps are completely uncorrelated

	saliencyMap1 and saliencyMap2 are 2 real-valued matrices

		Computer CC score .
		:param sal_map_1 : first saliency map
		:param sal_map_2 : second  saliency map.
		:return score: float : score

	"""
	if isinstance(sal_map_1, np.ndarray):
		sal_map_1 = np.array(sal_map_1, dtype=np.float32)
	elif sal_map_1.dtype != np.float32:
		sal_map_1 = sal_map_1.astype(np.float32)

	if isinstance(sal_map_2, np.ndarray):
		sal_map_2 = np.array(sal_map_2, dtype=np.float32)
	elif sal_map_1.dtype != np.float32:
		sal_map_2 = sal_map_2.astype(np.float32)

	if sal_map_1.size != sal_map_2.size:
		sal_map_1 = imresize(sal_map_1, sal_map_2.shape)

	sal_map_1 = (sal_map_1 - sal_map_1.mean()) / (sal_map_1.std())
	sal_map_2 = (sal_map_2 - sal_map_2.mean()) / (sal_map_2.std())

	score = np.corrcoef(sal_map_1,sal_map_2)

	return score


def EMD():
	"""

	if you are using this function, please cite the following papers:

		http://www.ariel.ac.il/sites/ofirpele/fastemd/code/
		https://github.com/wmayner/pyemd

	"""
	pass




def KLdiv(saliency_map , fixation_map):
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

	if isinstance(fixation_map, np.ndarray):
		fixation_map = np.array(fixation_map, dtype=np.float32)
	elif fixation_map.dtype != np.float32:
		fixation_map = fixation_map.astype(np.float32)

	# the function will normalize maps before computing Kld
	score = entropy(saliency_map, fixation_map)
	return score



def AUC(salMap, fixMap):
	"""Computes AUC for given saliency map 'salMap' and given
	fixation map 'fixMap'
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

	fixMap = (fixMap>0.7).astype(int)
	salShape = salMap.shape
	fixShape = fixMap.shape

	predicted = salMap.reshape(salShape[0]*salShape[1], -1, order='F').flatten()
	actual = fixMap.reshape(fixShape[0]*fixShape[1], -1, order='F').flatten()
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

	saliency_map = fixation_map - np.min(fixation_map)
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
	return scrore



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



def MultiMatch(matlab_engine, P, Q, check=False, **kwargs):
	"""
		works only if you have matlab & matlab API installed

		1 )
			cd MATLAB_ROOT/extern/engines/python/
			sudo python setup.py install
		2 )
			Please download MultiMatch from the following link and
			extract it in  metric directory

			wget http://dev.humlab.lu.se/www-transfer/people/marcus-nystrom/MultiMatchToolbox.zip
			unzip MultiMatchToolbox.zip -d metrics && rm MultiMatchToolbox.zip


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

		result = matlab_engine.doComparison(P,Q, stdout=StringIO())
		return np.array(result).squeeze()
	except Exception as e:
		print(e)
		return



