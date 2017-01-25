"""
Vehicle classification models and features.
"""

from . import config

import numpy as np
np.random.seed(1337)

from skimage import io, feature, color
from glob import glob
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, ClassifierMixin

class ImageFeatExtractor(BaseEstimator):
	"""Feature extractor for vehicle/non-vehicle images.
	Two types of features used: 
		1. hog of gray 
		2. color histogram (to reduce false positive?)
	"""
	def __init__(self, pixels_per_cell=(8,8), 
		cells_per_block=(2,2), hist_nbins=32, **kwargs):
		# for hog features
		self.pixels_per_cell = pixels_per_cell
		self.cells_per_block = cells_per_block
		# for color histogram features
		self.hist_nbins = hist_nbins
		# feature extractors
		self.extract_hog = lambda img: feature.hog(color.rgb2gray(img), 
				pixels_per_cell=self.pixels_per_cell,
				cells_per_block=self.cells_per_block)
		self.extract_hist = lambda img, chann: np.histogram(img[:,:,chann],
				bins=self.hist_nbins, range=(0, 256))[0].astype(np.float)
		# feature standarizer
		self.hog_ss = StandardScaler()
		self.rhist_ss = StandardScaler()
		self.ghist_ss = StandardScaler()
		self.bhist_ss = StandardScaler()
	def fit_transform(self, images, labels=None):
		hog_feats = np.vstack([self.extract_hog(im) for im in images])	
		hog_feats = self.hog_ss.fit_transform(hog_feats)

		rhist_feats = np.vstack([self.extract_hist(im, 0) for im in images])
		rhist_feats = self.rhist_ss.fit_transform(rhist_feats)

		ghist_feats = np.vstack([self.extract_hist(im, 1) for im in images])
		ghist_feats = self.ghist_ss.fit_transform(ghist_feats)

		bhist_feats = np.vstack([self.extract_hist(im, 2) for im in images])
		bhist_feats = self.bhist_ss.fit_transform(bhist_feats)

		return np.hstack([hog_feats])

		return np.hstack([hog_feats, rhist_feats, ghist_feats, bhist_feats])

	def fit(self, images, labels=None):
		self.fit_transform(images)
		return self
	def transform(self, images, labels=None):
		hog_feats = np.vstack([self.extract_hog(im) for im in images])
		hog_feats = self.hog_ss.transform(hog_feats)

		rhist_feats = np.vstack([self.extract_hist(im, 0) for im in images])
		rhist_feats = self.rhist_ss.transform(rhist_feats)

		ghist_feats = np.vstack([self.extract_hist(im, 1) for im in images])
		ghist_feats = self.ghist_ss.transform(ghist_feats)

		bhist_feats = np.vstack([self.extract_hist(im, 2) for im in images])
		bhist_feats = self.bhist_ss.transform(bhist_feats)

		return np.hstack([hog_feats])

		return np.hstack([hog_feats, rhist_feats, ghist_feats, bhist_feats])

class VehicleClassifier(BaseEstimator, ClassifierMixin):
	"""Vehicle detection model based on HOG feature and 
	Linear SVC model.
	"""
	def __init__(self, **kwargs):
		# parameter for LinearSVC
		self.C = kwargs.get("C", 1)
		self.model = Pipeline([
			("feature_extractor", ImageFeatExtractor(**kwargs)),
			("svc", LinearSVC(C=self.C))
			# ("svc", SVC(C=self.C, kernel="linear", probability=True))
		])
	def fit(self, images, labels):
		"""`images`: list of images of fixed shape
		`labels`: list of labels, with the same len of images
		Return: a trained vehicle detection model
		"""
		
		self.model.fit(images, labels)
		return self
	def best_fit(self, images, labels):
		"""Find best models by random search
		"""
		params = {
			'svc__C': np.logspace(-3, 3, 10),
			'feature_extractor__pixels_per_cell': [(6, 6), (8, 8)],
			'feature_extractor__cells_per_block': [(2, 2), (3, 3)],
			'feature_extractor__hist_nbins':[32, 64]
		}
		searcher = RandomizedSearchCV(self.model, 
			params, 
			cv=3, n_jobs=1, verbose=2, n_iter=15)
		searcher.fit(images, labels)
		self.model = searcher.best_estimator_
		self.best_params = searcher.best_params_
		return self
	def predict(self, images):
		return self.model.predict(images)
	def score(self, images, labels):
		return self.model.score(images, labels)


def load_data():
	vehicle_files = glob(config.vehicle_image_files)
	nonvehicle_files = glob(config.nonvehicle_image_files)
	print("loaded %i vehicle images and %i nonvehicle images" % 
		(len(vehicle_files), len(nonvehicle_files)))
	vehicle_imgs = [io.imread(f) for f in vehicle_files]
	nonvehicle_imgs = [io.imread(f) for f in nonvehicle_files]
	images = vehicle_imgs + nonvehicle_imgs
	labels = ["vehicle"]*len(vehicle_imgs)+["nonvehicle"]*len(nonvehicle_imgs)
	images, labels = shuffle(images, labels)
	return images, labels

def fit_best_model():
	images, labels = load_data()
	train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2)
	classifier = VehicleClassifier()
	classifier.best_fit(train_images, train_labels)
	print("best model performance on test:", classifier.score(test_images, test_labels))
	return classifier

def build_model():
	images, labels = load_data()
	train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2)
	classifier = VehicleClassifier(
			pixels_per_cell=(8,8),
			cells_per_block=(2,2),
			hist_nbins=32,
			C=5e-3)
	classifier.fit(train_images, train_labels)
	print("built model performance on test:", classifier.score(test_images, test_labels))
	return classifier