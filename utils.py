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
		ybin = ((height - fixation[1])//height_step)
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




# def MM_simplify_scanpath(P, th_glob, th_dur, th_amp):

# 	class Scanpath(object):
# 		"""
# 			Modeling scanpaths simialar to codes published by authors.

# 		"""
# 		def __init__(self, fixation==list()):
# 			# class Saccade(object):
# 				# def __init__(self, fixations):
# 			self.x = list()
# 			self.y = list()
# 			self.lenx = list()
# 			self.leny = list()
# 			self.len = list()
# 			self.theta = list()
# 			self.dur = list()
# 			if fixations:
# 				self.prep(fixations)

# 		def prep(self, fixations):
# 			for fix_idx, fix in enumerate(fixations):
# 				self.x.append(fix[0])
# 				self.y.append(fix[1])
# 				self.dur.append(fix[2])
# 				if fix_idx >= 1:
# 					self.lenx.append(fix[0] - self.x[fix_idx -1])
# 					self.leny.append(fix[1] - self.y[fix_idx -1])
# 					tmp = self.cart2pol(self.lenx[fix_idx-1], self.leny[fix_idx-1])
# 					self.theta.append(tmp[1])
# 					self.len.append(tmp[0])

# 		def cart2pol(self, x, y):
# 			rho = np.sqrt(x**2 + y**2)
# 			phi = np.arctan2(y, x)
# 			return(rho, phi)


# 		def add_saccade(self, x, y, lenx, leny, Len, theta, dur):
# 			self.x.append(x)
# 			self.y.append(y)
# 			self.lenx.append(lenx)
# 			self.leny.append(leny)
# 			self.len.append(Len)
# 			self.theta.append(theta)
# 			self.dur.append(dur)


# 	def simplify_duration(P, th_glob=th_glob, th_dur=th_dur):
# 		i = 0

# 		p_sim = Scanpath()

# 		while i < len(P.x):
# 			if i < length(sp.saccade.x):
# 				v1=[P.lenx[i],P.leny[i]];
# 				v2=[P.lenx[i+1],P.leny[i+1]];
# 				angle = np.arccos(np.dot(v1,v2))
# 				angle = angle / (np.linalg.norm(v1,2)*np.linalg.norm(v2,2));
# 			else:
# 				angle = np.inf;

# 			if (angle < th_glob) and (i < len(P.x)):

# 				#Do not merge saccades if the intermediate fixation druations are
# 				# long
# 				if P.dur[i+1] >= th_dur:
# 					p_sim.add_saccade()
# 					[sp,spGlobal,i,durMem] = keepSaccade(sp,spGlobal,i,j,durMem);
# 					j = j+1;
# 					continue,
# 				end

# 				% calculate sum of local vectors.
# 				v_x = sp.saccade.lenx(i) + sp.saccade.lenx(i+1);
# 				v_y = sp.saccade.leny(i) + sp.saccade.leny(i+1);
# 				[theta,len] = cart2pol(v_x,v_y);

# 				% ... save them a new global vectors
# 				spGlobal.saccade.x(j) = sp.saccade.x(i);
# 				spGlobal.saccade.y(j) = sp.saccade.y(i);
# 				spGlobal.saccade.lenx(j) = v_x;
# 				spGlobal.saccade.leny(j) = v_y;
# 				spGlobal.saccade.len(j) = len;
# 				spGlobal.saccade.theta(j) = theta;

# 				%... and sum up all the fixation durations
# 				spGlobal.fixation.dur(j) = sp.fixation.dur(i);%+sp.fixation.dur(i+1)/2+durMem;
# 				durMem = 0;%sp.fixation.dur(i+1)/2;
# 				i = i+2;


# 	def simplyfy_length(th_glob=th_glob, th_amp=th_amp):
# 		pass


# 	P = scanpath(P)

# 	l = 10000
# 	while True
# 		P = simplify_duration(P, Tdur)
# 		P = simplify_length(P, Tamp)
# 		if l == P.fixation[0]:
# 			break
# 		l = len(P.fixation[0])

# 	return P









