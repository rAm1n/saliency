from .metrics import *

STATIC_MTERICS = [
	AUC,
	SAUC,
	NSS,
	CC,
	KLdiv,
	IG,
	SIM,
	EMD,
]


SEQUENTIAL_METRICS = [
	euclidean_distance,
	mannan_distance,
	eyenalysis,
	levenshtein_distance,
	scan_match,
	hausdorff_distance,
	frechet_distance,
	DTW,
	TDE,
	REC,
	DET,
	LAM,
	CORM,
	multi_match
]


