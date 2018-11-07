import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
import urllib.request
import os
import zipfile


def main():
	#step1 : download google's pre-trained neural network
	url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
	data_dir='../data'
	model_name = os.path.split(url)[-1]
	local_zipfile= os.path.join(data_dir,model_name)
	if not os.path.exists(local_zipfile):
		#download
		model_url = urllib.request.urlopen(url)
		with open(local_zipfile, 'wb') as output:
			output.write(model_url.read())
		#extract
		with zipfile.ZipFile(local_zipfile,'r') as zip_ref:
			zip_ref.extractall(data_dir)


	model_fn = 'tensorflow_inception_graph.pb'


	#step 2 : Creating tensorflow session and loading the model

	graph = tf.Graph()
	sess = tf.InteractiveSession(graph=graph)
	with tf.gfile.FastGFile(os.path.join(data_dir,model_fn),'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	t_input = tf.placeholder(np.float32, name='input')
	imagenet_mean = 117.0
	t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
	tf.import_graph_def(graph_def, {'input': t_preprocessed})
	layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
	