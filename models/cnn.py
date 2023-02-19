import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import kerastuner as kt

class ConvNN(kt.HyperModel):
	def __init__(self,batch_size=32, nb_labels=2, epochs=50, weights=None):
		self.batch_size = batch_size
		print(self.batch_size)
		self.nb_labels = nb_labels
		self.epochs = epochs
		self.weights = weights
		self.input_shape = [224, 224, 3]

	def setup(self, units=256, dropout_rate=0.5, lr=1e-4):
		# Input shape = (None,1,224,224,3)
		inputs = Input(shape=self.input_shape)
		#define data augmentation, note this layer These layers are active only during training
		data_augmentation = Sequential([
  							RandomFlip('horizontal'),
							RandomFlip('vertical'),
  							RandomRotation(0.2),
							])
		#rescale the input value to this model expected pixel values in [-1, 1]
		preprocessed_inputs = mobilenet_v2.preprocess_input(inputs)
		# use pretrained MobileNet V2 
		base_model = MobileNetV2(input_shape=self.input_shape,
											   include_top=False,
											   pooling='avg',
											   weights='imagenet')
		# freeze the convolution base
		base_model.trainable = True
		
		# get the prediction layer
		prediction_layer = Dense(self.nb_labels)

		# build the model
		x = data_augmentation(inputs)
		x = mobilenet_v2.preprocess_input(x)
		x = base_model(x, training=False)
		x = Dense(units=units)(x)
		x = Dropout(dropout_rate)(x)
		# x = Dense(256, activation='relu', name='hidden_layer')(x)
		# x = Dropout(0.5)(x)
		outputs = Dense(self.nb_labels, activation='sigmoid', name='outputs')(x)
		model = Model(inputs, outputs)
		print(model.summary())
	
		# Define the optimizer learning rate as a hyperparameter.
		adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		model.compile(loss=WeightedMacroSoftF1(self.weights),
					  optimizer=adam,
					  metrics=[macro_f1])
		self.model = model
		return model

	def parameter_search(self, hp):
		units = hp.Int("units", min_value=128, max_value=512, step=64)
		dropout_rate = hp.Float('dropout', 0, 0.5, step=0.1, default=0.5)
		lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
		# call existing model-building code with the hyperparameter values.
		model = self.setup(units=units, dropout_rate=dropout_rate, lr=lr)
		return model
	
	def fit(self, X_train,Y_train,X_val, y_val):
		# early_stop = MyEarlyStopping(patience=20, verbose=0)
		early_stop = EarlyStopping(monitor="val_macro_f1", patience=20, verbose=0, mode="max")
		checkpointer = ModelCheckpoint(
			filepath="weights_best.h5",
			verbose=0, save_best_only=True)
		
		self.model.fit(X_train, Y_train, batch_size=self.batch_size,
						epochs=self.epochs, validation_data=(X_val,y_val),
						callbacks=[early_stop,checkpointer], verbose=2)
	
	def load_trained_weights(self, filename):
		self.model.load_weights(filename)
		print ('Loading pre-trained weights from %s.' %filename)
		return self

	def evaluate(self, X, y, feature_names, save_dir):
		y_pred = self.model.predict(X, verbose=0)
		# post processing
		y_pred = self.post_process_pred(y_pred)
		tp, fp, fn, f1, shape_acc, ori_acc, avg_shape_f1, avg_ori_f1, avg_total_f1 = self.calculate_f1(y, y_pred)
		results = self.write_metrics_to_json(tp, fp, fn, f1, shape_acc, ori_acc, avg_shape_f1,
									 avg_ori_f1, avg_total_f1, feature_names, save_dir)
		print(results)

	def predict(self, X, img_names, feature_names, save_dir):
		y_pred = self.model.predict(X, verbose=0)
		y_pred = self.post_process_pred(y_pred)
		pred = self.get_shape_orientation(y_pred, feature_names)
		self.write_preds_to_json(img_names, pred, save_dir)

	def write_preds_to_json(self, img_names, shapes_and_orientations, save_dir):
		# Combine the two arrays into a list of tuples
		data = []
		for img_name, shape_and_orientation in zip(img_names, shapes_and_orientations):
			shape, orientation = shape_and_orientation
			data.append({'img_names': img_name, 'shape': shape, 'orientation': orientation})

		# Write the list of tuples as a JSON object to a file
		with open(os.path.join(save_dir,'predictions.json'), 'w') as outfile:
			json.dump(data, outfile)
		

	def write_metrics_to_json(self, tp, fp, fn, f1, shape_acc, ori_acc,
								avg_shape_f1, avg_ori_f1, avg_total_f1, feature_names, save_dir):
		# Create a dictionary to store the results
		results = {
			'metrics': feature_names,
			'TP': [int(value) for value in tp],
			'fp': [int(value) for value in fp],
			'fn': [int(value) for value in fn],
			'f1': [round(value, 2) for value in f1],
			'shape acc': round(shape_acc, 2),
			'ori acc':  round(ori_acc, 2),
			'average shape f1': round(avg_shape_f1,2),
			'average ori f1': round(avg_ori_f1,2),
			'average total': round(avg_total_f1,2)
		}
		#Write the results to a JSON file
		with open(os.path.join(save_dir, 'eval_metric_results.json'), 'w') as f:
			json.dump(results, f, indent=4)
		return results

	def calculate_f1(self, y, y_pred):
		tp = np.sum(y_pred * y, axis=0)
		fp = np.sum(y_pred * (1 - y), axis=0)
		fn = np.sum((1 - y_pred) * y, axis=0)
		f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
		shape_acc = np.sum(tp[:3]) / len(y)
		ori_acc = np.sum(tp[-2:]) / len(y)
		avg_f1_shape = np.mean(f1[:3])
		avg_f1_ori = np.mean(f1[-2:])
		avg_f1 = np.mean([avg_f1_shape, avg_f1_ori])
		return tp, fp, fn, f1, shape_acc, ori_acc, avg_f1_shape, avg_f1_ori, avg_f1

	#predict post process
	def post_process_pred(self, y_pred):
		# Find the largest element in the shape feature and make it to 1
		max_shape_value = np.max(y_pred[:,:3],axis=1, keepdims=True)
		y_shape_feature = np.where(y_pred[:,:3] == max_shape_value, 1, 0).astype(np.int8)
		# Find the largest element in the ori feature
		max_ori_value = np.max(y_pred[:,-2:],axis=1, keepdims=True)
		y_ori_feature = np.where(y_pred[:,-2:] == max_ori_value, 1, 0).astype(np.int8)
		# Concatenate the two arrays along axis 1 to get the final predict
		y_pred = np.concatenate([y_shape_feature, y_ori_feature], axis=1)
		return y_pred

	# get the shape and orientation
	def get_shape_orientation(self, y, names):
		# loop through each row and create a new array with only the elements that are 1
		new_arr = []
		for i in range(y.shape[0]):
			indices = np.where(y[i] == 1)[0]
			new_row = [names[j] for j in indices]
			new_arr.append(new_row)
		return new_arr

	def unittest(self, X_val):
		preprocessed_inputs = mobilenet_v2.preprocess_input(X_val)
		print(np.amax(X_val), np.amin(X_val))
		print(np.amax(preprocessed_inputs), np.amin(preprocessed_inputs))

#Compute the macro soft F1-score as a cost (weighted 1 - soft-F1 across all labels).
# Use probability values instead of binary predictions
class WeightedMacroSoftF1(tf.keras.losses.Loss):
	def __init__(self, weights):
		super(WeightedMacroSoftF1, self).__init__()
		self.weights = weights
	@tf.function
	def call(self, y_true, y_pred):
		y = tf.cast(y_true, tf.float32)
		y_hat = tf.cast(y_pred, tf.float32)
		tp = tf.reduce_sum(y_hat * y, axis=0)
		fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
		fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
		soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
		cost = 1 - soft_f1
		weighted_cost = cost * self.weights
		# shape feature average weighted loss
		shape_avg_weighted_cost = tf.reduce_mean(weighted_cost[:3], axis=0)
		# orientation feature average weighted loss
		ori_avg_weighted_cost = tf.reduce_mean(weighted_cost[-2:], axis=0)
		# get the final average value of the two feature cost
		macro_cost = tf.divide(tf.add(shape_avg_weighted_cost, ori_avg_weighted_cost), 2.0)
		return macro_cost


@tf.function
def macro_f1(y, y_hat):
	"""Compute the macro F1-score on a batch of observations (average F1 across labels)
	"""
	#['oval', 'round', 'irregular', 'parallel', 'not_parallel']
	# Pick the largest one from the first three and pick the largest one from the "parallel" and "not_parallel"
	y = tf.cast(y, tf.float32)
	y_pred = tf.cast(y_hat, tf.float32)
	# Find the largest element in the shape feature and make it to 1
	max_shape_value = tf.reduce_max(y_pred[:,:3],axis=1)
	max_shape_value = tf.expand_dims(max_shape_value, axis=1)
	y_shape_feature = tf.cast(tf.equal(y_pred[:,:3], max_shape_value), dtype=tf.float32)
	# Find the largest element in the ori feature
	max_ori_value = tf.reduce_max(y_pred[:,-2:],axis=1)
	max_ori_value = tf.expand_dims(max_ori_value, axis=1)
	y_ori_feature = tf.cast(tf.equal(y_pred[:,-2:], max_ori_value), dtype=tf.float32)
	# Concatenate the two tensors along axis 0 to get the final predict
	y_pred = tf.concat([y_shape_feature, y_ori_feature], axis=1)
	tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
	fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
	fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
	f1 = 2*tp / (2*tp + fn + fp + 1e-16)
	# shape feature average f1
	shape_f1 = tf.reduce_mean(f1[:3], axis=0)
	# orientation feature average f1
	ori_f1 = tf.reduce_mean(f1[-2:], axis=0)
	# get the final f1
	macro_f1= tf.divide(tf.add(shape_f1, ori_f1), 2.0)
	return macro_f1
	
