#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from glob import glob
import cv2
import matplotlib.pyplot as plt


class ReadData:
    def __init__(self, img_dir, anno_dir):
        self.img_dir = img_dir
        self.anno_dir = anno_dir

        self.images, self.shape_labels, \
        self.orientation_labels, \
        self.image_locations = self.read_data(self.img_dir, self.anno_dir)

        self.ori_label_dic = self.get_label_dic(self.orientation_labels)
        self.shape_label_dic = self.get_label_dic(self.shape_labels)
    
    @staticmethod
    def get_label_dic(label_li):
        label_label_dic = {}
        for diff_label in label_li:
            label_label_dic[diff_label] = label_li.count(diff_label)
        return label_label_dic

    @staticmethod
    def read_data(image_folder_location, annotations_location):
        """
        Created on 10/2/23 2:54 pm

        @author: melih
        """
        image_locations = glob(image_folder_location + "/*.png")
        with open(annotations_location) as f:
            annotations = json.load(f)

        images = []
        shape_labels = []
        orientation_labels = []

        for loc in image_locations:
            im = cv2.imread(loc)
            if len(im.shape) == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            else:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            image_name = loc.split("/")[-1]
            shape_label = annotations[image_name]["shape"]
            orientation_label = annotations[image_name]["orientation"]
            images.append(im)
            shape_labels.append(shape_label)
            orientation_labels.append(orientation_label)

        return images, shape_labels, orientation_labels, image_locations
        # images, shape_label, orientation_label, image_locations = read_data("tech_task_data",
#                                                                     "annotations.json")

    def plot_data(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax1.bar(list(self.ori_label_dic.keys()), list(self.ori_label_dic.values()), width = 0.3)
        ax1.set_title('Shape feature')
        ax1.set_ylim([0, len(self.shape_labels)])
        ax1.bar_label(ax1.containers[0], label_type='edge')
        fig.tight_layout(pad=2.5)

        ax2 = fig.add_subplot(2,1,2)
        ax2.bar(list(self.shape_label_dic.keys()), list(self.shape_label_dic.values()),width = 0.6)
        ax2.set_title('orientation feature')
        ax2.set_ylim([0, len(self.orientation_labels)])
        ax2.bar_label(ax2.containers[0], label_type='edge')
        # Save the full figure...
        fig.savefig('Feature.png')

    @staticmethod
    def get_img_sz_range(images):
        width =[1e5,0]
        height =[1e5,0]
        for img in images:
            width[0] = min(width[0], img.shape[0])
            width[1] = max(width[1], img.shape[0])
            height[0] = min(height[0], img.shape[1])
            height[1] = max(height[1], img.shape[1])
        print('width', width)
        print('height', height)


if __name__ == '__main__':
    image_folder_location = 'data/tech_task_data' 
    annotations_location = 'data/annotations.json'

    dataset = ReadData(image_folder_location, annotations_location)
    images, shape_label, orientation_label, image_locations = dataset.read_data(image_folder_location, annotations_location)

    dataset.get_img_sz_range(images)

    # dataset.plot_data()
    print(dataset.ori_label_dic)
    print(dataset.shape_label_dic)

    # plot_data(shape_label)
    print('total images number:', len(images))
    print('total irregular number', shape_label.count('irregular'))
    print('total oval number', shape_label.count('oval'))
    print('total round number', shape_label.count('round'))
    print('total parallel number', orientation_label.count('parallel'))
    print('total not_parallel number', orientation_label.count('not_parallel'))

