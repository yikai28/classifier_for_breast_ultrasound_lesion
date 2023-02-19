import json
from glob import glob
import cv2
import copy
# from read_data import ReadData
import numpy as np
from sklearn.preprocessing import OneHotEncoder
# from keras.utils.np_utils import to_categorical   
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import albumentations as A

class PrepareData:
    def __init__(self, imgs, shape_labels, orientation_labels, img_locations, resize_width=224, resize_height=224):
        self.imgs = imgs
        self.img_locations = img_locations
        self.shape_labels = shape_labels
        self.orientation_labels = orientation_labels
        self.labels = [[self.shape_labels[i], self.orientation_labels[i]] for i in range(len(self.shape_labels))]
        self.resize_width = resize_width
        self.resize_height = resize_height

        self.mlb = MultiLabelBinarizer(classes=['oval', 'round', 'irregular', 'parallel', 'not_parallel'])
        self.get_unique_label()
        self.get_label_cats()

        # set the resize transformation
        self.resize =  A.Compose([
                A.LongestMaxSize(max_size=max(self.resize_height,self.resize_width), interpolation=1),
                A.PadIfNeeded(min_height=self.resize_height, min_width=self.resize_width, border_mode=0, value=(0,0,0)),])
        # Specify data augmentations here
        self.transformations = A.Compose([
                A.OneOf([
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                ]), 
        ])

    def get_unique_label(self):
        # get the caterogy type
        self.cats_dic_default = {}
        unique_shape_labels = set(self.shape_labels)
        unique_orientation_labels = set(self.orientation_labels)
        for shape_label in unique_shape_labels:
            for orientation_label in unique_orientation_labels:
                self.cats_dic_default.setdefault(tuple([shape_label, orientation_label]), 0)
    
    def get_label_cats(self):
        self.cats_dic={}
        cat_id = 0
        for key in self.cats_dic_default.keys():
            self.cats_dic[key] = cat_id
            cat_id +=1
        print(self.cats_dic)

    @staticmethod
    def normalize(pixels):
        mean, std = pixels.mean(), pixels.std()
        pixels = (pixels - mean) / std
        pixels = np.clip(pixels, -1.0, 1.0)
        pixels = (pixels + 1.0) / 2.0   

        return pixels

    @staticmethod
    def count_label(self, cats_dic_default):
        for label in self.labels:
            cats_dic_default[tuple(label)] +=1
        print(cats_dic_default)
        # get each label number

    def covert_onehot_label(self, labels):
        labels = self.mlb.inverse_transform(labels)
        labels_encode = [self.cats_dic[tuple(label)] for label in labels]
        # self.y = to_categorical(self.labels_encode, num_classes=6)
        return labels_encode
      

    def train_val_test_split(self):
        # multi-label stratification methods to split train, val, test 0.8, 0.1, 0.1
        mlb_label = self.mlb.fit_transform(self.labels)
        feature_names = list(self.mlb.classes_)
        print(feature_names)
        # X = np.reshape(np.array(self.imgs), (len(np.array(self.imgs)), 1))
        X = np.expand_dims(np.array(self.imgs), axis=1)
        y = np.array(mlb_label)
        # get the different features number
        feature_nums = np.count_nonzero(y, axis=0)
        #calculate the different features number dictionary
        feature_dic = {}
        for feature_name, feature_num in zip(feature_names, feature_nums):
            feature_dic[feature_name] = feature_num
        #set the random seeds
        random_state=42
        np.random.seed(random_state)
        #split the data to train, val, test 0.8, 0.1, 0.1
        X_train, y_train, X_val_test, y_val_test = iterative_train_test_split(X, y, 0.2)
        X_val, y_val, X_test, y_test = iterative_train_test_split(X_val_test, y_val_test, 0.5)
        X_train, X_val, X_test = np.squeeze(X_train), np.squeeze(X_val), np.squeeze(X_test)

        # change all the image to the same size 
        X_train = np.array([self.resize(image=img)["image"] for img in X_train ])
        X_val = np.array([self.resize(image=img)["image"] for img in X_val ])
        X_test = np.array([self.resize(image=img)["image"] for img in X_test ])

        return X_train, y_train, X_val, y_val, X_test, y_test, feature_dic

if __name__ == '__main__':
    from read_data import ReadData
    image_folder_location = 'data/tech_task_data' 
    annotations_location = 'data/annotations.json'
    dataset = ReadData(image_folder_location, annotations_location)
    print(dataset.image_locations[:10])
    data_prepare = PrepareData(dataset.images, dataset.shape_labels, dataset.orientation_labels, dataset.image_locations, 
                                resize_width=224, resize_height=224)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_dic = data_prepare.train_val_test_split()
    
    print(X_train.shape, X_val.shape, X_test.shape)
    print(y_train.shape, y_val.shape, y_test.shape)
    print(feature_dic)



    # print(y_val[0], y_test[0])
    # The usage for the data transformation
    # transformed = data_prepare.transformations(image=X_train[0])
    # cv2.imwrite('original.png',X_train[0])
    # cv2.imwrite('transform.png',transformed["image"])
    # print(X_train[0].shape, transformed["image"].shape)
    # print(np.array_equal(X_train[0], transformed["image"]))
    













    # print(X_val[0].shape)
    # print(len(y_train), len(y_val), len(y_test))
    # print(np.sum(y_train,axis=0),np.sum(y_val,axis=0),np.sum(y_test,axis=0))
    #['irregular', 'not_parallel', 'oval', 'parallel', 'round']
    # oval - parallel
    # y_all = [y_train, y_val, y_test]
    # for y in y_all:
    #     print(np.count_nonzero(np.all(y == [0, 0, 1, 1, 0], axis=1)))

    #     # oval - non-parallel
    #     print(np.count_nonzero(np.all(y == [0, 1, 1, 0, 0], axis=1)))

    #     # round, parallel
    #     print(np.count_nonzero(np.all(y == [0, 0, 0, 1, 1], axis=1)))

    #     # round, non-parallel
    #     print(np.count_nonzero(np.all(y == [0, 1, 0, 0, 1], axis=1)))

    #     # irregular, parallel
    #     print(np.count_nonzero(np.all(y == [1, 0, 0, 1, 0], axis=1)))

    #     # irregular, non-parallel
    #     print(np.count_nonzero(np.all(y == [1, 1, 0, 0, 0], axis=1)))

    #     print('----------------------------------------------------')



