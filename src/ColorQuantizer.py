#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ColorQuantizer.py
#
# Created by Samet Cetin.
# Contact: cetin.samet@outlook.com
#

import numpy as np
from scipy.ndimage import imread
from scipy.misc import imsave
from MyKMeans import MyKMeans


class ColorQuantizer:
    """Quantizer for color reduction in images. Use MyKMeans class that you implemented.
    
    Parameters
    ----------
    n_colors : int, optional, default: 64
        The number of colors that wanted to exist at the end.
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Read more from:
    http://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
    """
    
    def __init__(self, n_colors=64, random_state=None):
        self.n_colors       = n_colors

        if type(random_state) == int:                       # CHECK IF random_state IS AN integer
            self.random_state   = np.random.RandomState(random_state)
        elif type(random_state) == np.random.RandomState:   # CHECK IF random_state IS A np.random.RandomState
            self.random_state   = random_state
        else:                                               # CHECK IF random_state IS None
            self.random_state = np.random
    
    def read_image(self, path):
        """Reads jpeg image from given path as numpy array. Stores it inside the
        class in the image variable.
        
        Parameters
        ----------
        path : string, path of the jpeg file
        """
        self.image = imread(fname=path)
        return
    
    def recreate_image(self, path_to_save):
        """Reacreates image from the trained MyKMeans model and saves it to the
        given path.
        
        Parameters
        ----------
        path_to_save : string, path of the png image to save
        """
        image_arr       = np.reshape(self.image, newshape=(-1, 3))
        sample_size     = 4000
        image_sample    = image_arr[self.random_state.permutation(image_arr.shape[0])[:sample_size]]

        kmeans = MyKMeans(n_clusters=self.n_colors, init_method='kmeans++')
        kmeans.initialize(image_sample)
        kmeans.fit(image_sample)
        self.cluster_centers = kmeans.cluster_centers

        predicted_labels = kmeans.predict(image_arr)

        quantized_image = np.array([kmeans.cluster_centers[label] for label in predicted_labels])
        quantized_image = np.reshape(quantized_image, self.image.shape).astype(np.uint8)

        imsave(path_to_save, quantized_image)
        return

    def export_cluster_centers(self, path):
        """Exports cluster centers of the MyKMeans to given path.

        Parameters
        ----------
        path : string, path of the txt file
        """
        np.savetxt(path, self.cluster_centers)
        return
        
    def quantize_image(self, path, weigths_path, path_to_save):
        """Quantizes the given image to the number of colors given in the constructor.
        
        Parameters
        ----------
        path : string, path of the jpeg file
        weigths_path : string, path of txt file to export weights
        path_to_save : string, path of the output image file
        """
        self.read_image(path=path)
        self.recreate_image(path_to_save=path_to_save)
        self.export_cluster_centers(path=weigths_path)
        return
        