import argparse
import json
import os
import numpy as np
from utils.read_data import ReadData
from utils.prep_data import PrepareData
from models.cnn import ConvNN
import keras_tuner as kt
from keras_tuner import Objective
import tensorflow as tf

def preprocessing(image_folder_location, annotations_location):
    dataset = ReadData(image_folder_location, annotations_location)
    data_prepare = PrepareData(dataset.images, dataset.shape_labels, dataset.orientation_labels, 
                                dataset.image_locations, resize_width=224, resize_height=224)        
    return data_prepare

def parameter_search(X_train, y_train, X_val, y_val, weights, hp):
    model = ConvNN(batch_size=32, nb_labels=5, epochs=30, weights=weights)
    objective = Objective('val_macro_f1', direction='max')
    tuner = kt.RandomSearch(
            hypermodel=model.parameter_search,
            objective=objective,
            max_trials=100,
            executions_per_trial=2,
            overwrite=True,
            directory="results",
            project_name="parameter_search",
            )
    tuner.search_space_summary()
    tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val), 
                callbacks=[tf.keras.callbacks.EarlyStopping('val_macro_f1', mode="max", patience=15)])
    # Get the best hyperparameters from the tuner
    best_hps = tuner.get_best_hyperparameters(1)[0]
    # Convert the best hyperparameters object to a dictionary
    best_hps_dict = best_hps.get_config()
    # Write the dictionary to a JSON file
    with open('best_hps.json', 'w') as f:
        json.dump(best_hps_dict, f)

def train(X_train, y_train, X_val, y_val, weights):
    model = ConvNN(batch_size=32,nb_labels=5,epochs=100,weights=weights)
    model.setup(units=448, dropout_rate=0.4, lr=1e-4)
    model.fit(X_train, y_train, X_val, y_val)

def test(X_test, y_test, weights, feature_names, best_model_path="weights_best.h5", eval_save_dir='results'):
    model = ConvNN(batch_size=X_test.shape[0],nb_labels=5,epochs=100,weights=weights)
    model.setup(units=448, dropout_rate=0.4, lr=1e-4)
    assert os.path.exists(best_model_path)
    model.load_trained_weights(best_model_path)
    model.evaluate(X_test, y_test, feature_names, eval_save_dir)

def predict(X, img_names, weights, feature_names, best_model_path="weights_best.h5", pred_save_dir='pred'):
    model = ConvNN(batch_size=len(img_names),nb_labels=5,epochs=100,weights=weights)
    model.setup(units=448, dropout_rate=0.4, lr=1e-4)
    print(X.shape)
    model.predict(X, img_names, feature_names, pred_save_dir)

def main():
    # define input args
    parser = argparse.ArgumentParser(
        description='Convert masks to a json annotation file.')
    parser.add_argument(
        "--mode", help="parameter_search, train, test, pred",
        type=str, required=False, default='parameter_search')
    parser.add_argument(
        "--img-dir", help="Directory of images.",
        type=str, required=False, default='data/tech_task_data')
    parser.add_argument(
        "--annotations", help="Input json annotation file",
        type=str, required=False, default='data/annotations.json')
    parser.add_argument(
        "--eval-save-dir", help="results save directory",
        type=str, required=False, default='results')
    parser.add_argument(
        "--pred-save-dir", help="extra predictions save directory",
        type=str, required=False, default='predictions')
    args = parser.parse_args()
    assert args.mode in ['parameter_search', 'train', 'test', 'pred']

    # preprocessing the data
    data_prepare = preprocessing(args.img_dir, args.annotations)
    # split the data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_dic = data_prepare.train_val_test_split()
    #features_name list
    feature_names = list(feature_dic.keys())
    # calculate the weight based on the num of features 
    weights = [round(100.0/value,2) for value in list(feature_dic.values())]
    # set the keras tuner for hypermeter selecting
    hp = kt.HyperParameters()

    if args.mode == 'parameter_search':
        os.makedirs(args.eval_save_dir, exist_ok=True)
        #parameter search
        parameter_search(X_train, y_train, X_val, y_val, weights, hp)
    if args.mode == 'train':
        # train the model
        train(X_train, y_train, X_val, y_val, weights)
    if args.mode == 'test':
        #make the directory if not exist
        os.makedirs(args.eval_save_dir, exist_ok=True)
        #test the model
        test(X_test, y_test, weights, feature_names, best_model_path="weights_best.h5", eval_save_dir=args.eval_save_dir)
    if args.mode == 'pred':
         #make the directory if not exist
        os.makedirs(args.pred_save_dir, exist_ok=True)
        X = np.array([data_prepare.resize(image=img)["image"] for img in data_prepare.imgs])
        img_names = [os.path.basename(location) for location in data_prepare.img_locations]
        predict(X, img_names, weights, feature_names, best_model_path="weights_best.h5", pred_save_dir=args.pred_save_dir)
        
if __name__ == '__main__':
    main()