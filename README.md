# Saliency Datasets and Metrics


> **On metrics for measuring scanpath similarity**<br>
> [Ramin Fahimi](https://www.raminfahimi.com), [Neil D.B Bruce](https://socs.uoguelph.ca/~brucen/)<br>
> https://link.springer.com/article/10.3758/s13428-020-01441-0


---



This repository contains an API for saliency prediction datasets along with most common evaluation metrics. This code will download required files from the website of the original publisher of datasets.



### **What do I need?**
 -   Python (2.7, 3.4+)
 -   Python package manager (pip)
 -   Matlab (optional - required for some of the metrics.)


### **Getting started**

 1. Clone the repository using the following command:

		`git clone git@github.com:rAm1n/saliency.git`

	or download a zip version from [master.zip](https://github.com/rAm1n/saliency/archive/master.zip)
2. Install required packages using pip
             `pip install -r requirements`

3. Follow the tutorial:  [help.ipynb](https://github.com/rAm1n/saliency/blob/master/help.ipynb)

### **Datasets**

At this moment, the following datasets are covered. I have plan to add more and complete this list. Some of them have other very useful annotations but given the variety of  types, I have decided to not include external information at this point.


|         Datasets       | Author  | paper                         |Extra note                      |
|----------------|---------------|----------------|-----------------------------|
|TORONTO| Neil Bruce, John K. Tsotsos. |  [Attention based on information maximization ](http://journalofvision.org/7/9/950/)            |           |
|CAT2000          |Ali Borji, Laurent Itti. | [CAT2000: A Large Scale Fixation Dataset for Boosting Saliency Research ](http://arxiv.org/abs/1505.03581)            ||
|CROWD          |Ming Jiang, Juan Xu, et al. | [Saliency in Crowd ](http://www.ece.nus.edu.sg/stfpage/eleqiz/publications/pdf/crowd_eccv14.pdf)            |            |
|KTH          |Gert Kootstra, Bart de Boer, et al.  | [Predicting Eye Fixations on Complex Visual Stimuli using Local Symmetry ](http://www.csc.kth.se/~kootstra/index.php?item=602&menu=&file=http://dx.doi.org/10.1007/s12559-010-9089-5)            |            |
|OSIE          |Juan Xu, Ming Jiang, et al. | [Predicting Human Gaze Beyond Pixels](http://www.ece.nus.edu.sg/stfpage/eleqiz/publications/pdf/saliency_jov14.pdf)            |Object level attributes - mouse tracking          |
|MIT1003          |Tilke Judd, Krista Ehinger, et al.  | [Learning to Predict where Humans Look](http://people.csail.mit.edu/tjudd/WherePeopleLook/Docs/wherepeoplelook.pdf)            |            |
|LOWRES          |Tilke Judd, Fredo Durand, et al. | [Fixations on Low-Resolution Images](http://www.journalofvision.org/content/11/4/14.full.pdf+html)            |           |
|PASCAL-S          |Yin Li , Xiaodi Hou , et al. | [The Secrets of Salient Object Segmentation](http://openaccess.thecvf.com/content_cvpr_2014/papers/Li_The_Secrets_of_2014_CVPR_paper.pdf)            |Segmentation masks from VOC10            |
|PASCAL-KYUN          |Kiwon Yun, Yifan Peng, et al. | [Studying Relationships Between Human Gaze, Description, and Computer Vision](https://www3.cs.stonybrook.edu/~kyun/papers/kiwon_cvpr13_gaze.pdf)            |Segmentation masks from VOC10    |
|SUN09          |Kiwon Yun, Yifan Peng, et al. | [Studying Relationships Between Human Gaze, Description, and Computer Vision](https://www3.cs.stonybrook.edu/~kyun/papers/kiwon_cvpr13_gaze.pdf)            |Segmentation masks from VOC10    |
|SALICON          |Ming Jiang, Shengsheng Huang, et al.| [SALICON: Saliency in Context](http://www-users.cs.umn.edu/~qzhao/publications/pdf/salicon_cvpr15.pdf) |Subset of MSCOCO          |
|EMOD          |S. Fan, Z. Shen, et al. | [Emotional Attention](https://nus-sesame.top/emotionalattention/) |emotion, object semantic categories, and high-level perceptual           |
|POET          | Dim P. Papadopoulos, et al. | [Pascal Objects Eye Tracking (POET) ](http://calvin.inf.ed.ac.uk/datasets/poet-dataset/)            |Segmentation masks from VOC10(TODO)            |


<!-- **Let's get started**: -->

<!-- A jupyter notebook version of this tutorial has been added:  [help.ipynb](https://github.com/rAm1n/saliency/blob/master/help.ipynb) -->

<!--

	"""
		Assuming that:

		N = Number of examples
		O = Number of observers
		F = Number of fixations
		D = Fixation spec (posX, posY, Duration, Start, End)

		* If the ordering is constant across images and represents specific observers,
		* it has been preserved (exp. OSIE).
	"""

    from saliency.dataset import SaliencyDataset
    dataset = SaliencyDataset()

	# Dataset files will be stored in ~/tmp/saliency.
	# Consider passing a config dict to change it accordingly
		CONFIG = {
			'data_path' : os.path.expanduser('~/tmp/saliency/'),
			'auto_download' : True,
			}
		dataset = SaliencyDataset(config=CONFIG)

	# get list of currenly converted datsets.
	dataset.dataset_names()
	['TORONTO', 'CAT2000', 'CROWD', 'SALICON', 'LOWRES',\
		 'KTH', 'OSIE', 'MIT1003', 'PASCAL-S', 'EMOD', 'POET',\
		 'PASCAL-KYUN', 'SUN09']

	# Load your favourite dataset.
   	 dataset.load('OSIE')

	# list of annotations available for the chosen dataset.
	 dataset.data_type
	>>> array([u'heatmap', u'sequence', u'sequence_mouse_lab', u'sequence_mouse_amt'], dtype='<U18')

	# from now on, 'get' function will be your friend.
	# It can retrieve and return scanpaths. use one of the keys from data_type
	sequence = dataset.get('sequence') # return a np array. (N,O,F,D)

	# Or only the path to heatmaps
	heatmap_path = dataset.get('heatmap_path') # will return a list of paths

	# Or it can read and return stimuli in numpy format
	stimuli = dataset.get('stimuli') # will return np array (N, W, H, 3)

	# Consider passing a list of index if not everything is needed.
	samples = dataset.get('stimuli_path', index=range(10))

	# Also a list of Users if you only need specific observers from the dataset
	seqs = dataset.get('sequence', index=range(100), users=range(2,10))

	# Some metrics like AUC need fixation maps instead of heatmaps (N, W, H)
	fixations = dataset.get('fixation', index=range(10))




### **Processing & filtering scanpaths.**

Eye-tracking data and specifically scanpath always have errors and out of boundary fixation points.

    # Fixations in percntile format according to image resolution (N,O,F,D)
    sequence = dataset.get('sequence', percentile=True)

	# Remove out of boundary fixations:
	sequence = dataset.get('sequence', percentile=True, modify='remove')
	# or bring them back right inside boundary.
	sequence = dataset.get('sequence', percentile=True, modify='fix')
 -->


### ** Evalulation and Metrics**


#### Static Metrics




|         Metrics       |Citation                          |                          
|----------------|-------------------------------|
|AUC | [Saliency and Human Fixations: State-of-the-Art and Study of Comparison Metrics](https://ieeexplore.ieee.org/document/6751253) |
|SAUC  | [SUN: A Bayesian framework for saliency using natural statistics](https://jov.arvojournals.org/article.aspx?articleid=2297284) |
|NSS          | [Components of bottom-up gaze allocation in natural scenes](https://jov.arvojournals.org/article.aspx?articleid=2132474) |          
|CC          | **Pearson's linear coefficient** |      
|KLdiv          | |   
|SIM          | |   
|IG       | [Information-theoretic model comparison unifies saliency metrics](https://www.pnas.org/content/112/52/16054)|   


#### Sequential Metrics



|    | Metric               | Origin |
|----|----------------------|----------|
| 1  | Euclidean distance   |          |
| 2  | Mannan distance      |  [The relationship between the locations of spatial features.](https://www.ncbi.nlm.nih.gov/pubmed/9061830)        |
| 3  | Eyeanalysis          | [A simple way to estimate similarity between pairs of eye movement](https://bop.unibe.ch/JEMR/article/view/2326)        |
| 4  | Levenshtein distance |    [Algorithms for defining visual regions-of-interest](https://ieeexplore.ieee.org/document/877520)      |
| 5  | ScanMatch            |  [ScanMatch: A Novel Method for Comparing Fixation Sequences.](https://seis.bristol.ac.uk/~psidg/ScanMatch/CMTG2010.pdf)        |
| 6  | Hausdorff distance   |    [Comparing images using the Hausdorff distance](https://ieeexplore.ieee.org/document/232073)      |
| 7  | Frechet distance     |   [Computing discrete Fréchet distance](http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf)       |
| 8  | Dynamic time warp    |   [Using dynamic time warping to find patterns in time series](http://www.aaai.org/Papers/Workshops/1994/WS-94-03/WS94-03-031.pdf)       |
| 9  | Time delay embedding | [Simulating human saccadic scanpaths on natural images](https://ieeexplore.ieee.org/document/5995423/) | Adopted from [Fixaton](https://github.com/dariozanca/FixaTons)|         |
| 10 | MultiMatch (5)       | [A Vector-based, Multidimensional Scanpath Similarity Measure.](https://dl.acm.org/citation.cfm?id=1743718)         |
| 11 | Recurrence           |  [Recurrence quantification analysis of eye movements](https://link.springer.com/article/10.3758/s13428-012-0299-5)       |
| 12 | Determinism          |  [Recurrence quantification analysis of eye movements](https://link.springer.com/article/10.3758/s13428-012-0299-5)        |
| 13 | Laminarity           |  [Recurrence quantification analysis of eye movements](https://link.springer.com/article/10.3758/s13428-012-0299-5)        |
| 14 | CORM                 |   [Recurrence quantification analysis of eye movements](https://link.springer.com/article/10.3758/s13428-012-0299-5)       |



**Note**: To make things run smoother, scanpaths has already been preprocessed and stored on dropbox. If you own one of the datasets and you don't like your data to be included in this package, please send a short message to  *fahimi72 At gmail* and it will be taken care of. we do not own any of the data and the rights belong to the original publisher of the datasets. Please make sure to cite the appropriate paper if you are using them.


