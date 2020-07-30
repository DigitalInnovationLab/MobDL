from forward import ForwardModel
import cv2
import numpy as np
from mrcnn.visualize import save_image
import os
from PIL import Image

frozen_graph_path = "frozen_model/mask_frozen_graph.pb"

# assign your own Config class
from mrcnn.config import Config

from tensorflow.python.client import device_lib

def get_available_gpus():
	local_device_protos = device_lib.list_local_devices()
	return [x.name for x in local_device_protos if x.device_type == 'GPU']


def load_model():
	class parasiteConfig(Config):
		"""Configuration for training on the toy  dataset.
        Derives from the base Config class and overrides some values.
        """
		# Give the configuration a recognizable name
		NAME = "test"
		GPU_COUNT = 1
		# We use a GPU with 12GB memory, which can fit two images.
		# Adjust down if you use a smaller GPU.
		IMAGES_PER_GPU = 1

		# Number of classes (including background)
		NUM_CLASSES = 2  # Background + parasite + egg

		IMAGE_MIN_DIM = 512
		IMAGE_MAX_DIM = 512
		IMAGE_PADDING = True

		# Skip detections with < 90% confidence
		DETECTION_MIN_CONFIDENCE = 0.85


	config = parasiteConfig()


	class InferenceConfig(parasiteConfig):
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1
		IMAGE_MIN_DIM = 512
		IMAGE_MAX_DIM = 512
		DETECTION_MIN_CONFIDENCE = 0.50
		NUM_CLASSES = 1 + 7
		BACKBONE = "resnet50"


	inference_config = InferenceConfig()
	print(os.path)

	forward_model = ForwardModel('/data/user/0/com.example.python_test/files/chaquopy/AssetFinder/app/mask_frozen_graph.pb', inference_config)

	return forward_model

def test_one_image(image, forward_model):
	images = np.expand_dims(image, axis=0)
	results = forward_model(images)
	save_image_dir = "/data/user/0/com.example.python_test/files/chaquopy/AssetFinder/app"
	file_name = "test"
	class_names = ['BG',
				   'red blood cell',
				   'trophozoite',
				   'difficult',
				   'ring',
				   'schizont',
				   'gametocyte',
				   'leukocyte']

	class_names_n = ['BG',
				   'nuclei']
	r = results[0]

	save_image(image, file_name, r['rois'], r['mask'], r['class'], r['scores'], class_names_n, filter_classs_names=None,
			   scores_thresh=0.5, save_dir=save_image_dir, mode=2)

	return os.path.join(save_image_dir,"{}.jpg".format(file_name))

def test(image_path, forward_model):
	test_image_path = image_path
	image = cv2.imread(test_image_path)
	r = test_one_image(image, forward_model)

	return r