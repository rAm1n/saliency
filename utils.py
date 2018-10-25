import numpy as np



def scanpath_to_string(scanpath, height, width, Xbins, Ybins, Tbins):
	"""
			a b c d ...
		A
		B
		C
		D

		returns Aa
	"""
	if Tbins !=0:
		try:
			assert scanpath.shape[1] == 3
		except Exception as x:
			print("Temporal information doesn't exist.")

	height_step, width_step = height//Ybins, width//Xbins
	string = ''
	num = list()
	for i in range(scanpath.shape[0]):
		fixation = scanpath[i].astype(np.int32)
		xbin = fixation[0]//width_step
		ybin = fixation[1]//height_step
		corrs_x = chr(65 + xbin)
		corrs_y = chr(97 + ybin)
		T = 1
		if Tbins:
			T = fixation[2]//Tbins
		for t in range(T):
			string += (corrs_y + corrs_x)
			num += [(ybin * Xbins) + xbin]
	return string, num


def global_align(P, Q, SubMatrix=None, gap=0, match=1, mismatch=-1):
	"""
		https://bitbucket.org/brentp/biostuff/src/
	"""
	UP, LEFT, DIAG, NONE = range(4)
	max_p = len(P)
	max_q = len(Q)
	score   = np.zeros((max_p + 1, max_q + 1), dtype='f')
	pointer = np.zeros((max_p + 1, max_q + 1), dtype='i')

	pointer[0, 0] = NONE
	score[0, 0] = 0.0
	pointer[0, 1:] = LEFT
	pointer[1:, 0] = UP

	score[0, 1:] = gap * np.arange(max_q)
	score[1:, 0] = gap * np.arange(max_p).T

	for i in range(1, max_p + 1):
		ci = P[i - 1]
		for j in range(1, max_q + 1):
			cj = Q[j - 1]
			if SubMatrix is None:
				diag_score = score[i - 1, j - 1] + (cj == ci and match or mismatch)
			else:
				diag_score = score[i - 1, j - 1] + SubMatrix[cj][ci]
			up_score   = score[i - 1, j] + gap
			left_score = score[i, j - 1] + gap

			if diag_score >= up_score:
				if diag_score >= left_score:
					score[i, j] = diag_score
					pointer[i, j] = DIAG
				else:
					score[i, j] = left_score
					pointer[i, j] = LEFT
			else:
				if up_score > left_score:
					score[i, j ]  = up_score
					pointer[i, j] = UP
				else:
					score[i, j]   = left_score
					pointer[i, j] = LEFT

	align_j = ""
	align_i = ""
	while True:
		p = pointer[i, j]
		if p == NONE: break
		s = score[i, j]
		if p == DIAG:
			# align_j += Q[j - 1]
			# align_i += P[i - 1]
			i -= 1
			j -= 1
		elif p == LEFT:
			# align_j += Q[j - 1]
			# align_i += "-"
			j -= 1
		elif p == UP:
			# align_j += "-"
			# align_i += P[i - 1]
			i -= 1
		else:
			raise ValueError
	# return align_j[::-1], align_i[::-1]
	return score.max()













