'''Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new process methods, etc...
'''
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import sys
import threading
import copy
import inspect
import types
import multiprocessing as mp

import keras.backend as K

def random_rotation(x, rg, row_index=1, col_index=2, channel_index=0,
                    fill_mode='nearest', cval=0., rng=None):
    theta = np.pi / 180 * rng.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_shift(x, wrg, hrg, row_index=1, col_index=2, channel_index=0,
                 fill_mode='nearest', cval=0., rng=None):
    h, w = x.shape[row_index], x.shape[col_index]
    tx = rng.uniform(-hrg, hrg) * h
    ty = rng.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_shear(x, intensity, row_index=1, col_index=2, channel_index=0,
                 fill_mode='nearest', cval=0., rng=None):
    shear = rng.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_zoom(x, zoom_range, row_index=1, col_index=2, channel_index=0,
                fill_mode='nearest', cval=0., rng=None):
    if len(zoom_range) != 2:
        raise Exception('zoom_range should be a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = rng.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_barrel_transform(x, intensity):
    # TODO
    pass


def random_channel_shift(x, intensity, channel_index=0, rng=None):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + rng.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def array_to_img(x, dim_ordering=K.image_dim_ordering(), mode=None, scale=True):
    from PIL import Image
    x = x.copy()
    if dim_ordering == 'th':
        x = x.transpose(1, 2, 0)
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[2] == 3 and mode == 'RGB':
        return Image.fromarray(x.astype('uint8'), mode)
    elif x.shape[2] == 1 and mode == 'L':
        return Image.fromarray(x[:, :, 0].astype('uint8'), mode)
    elif mode:
        return Image.fromarray(x, mode)
    else:
        raise Exception('Unsupported array shape: ', x.shape)


def img_to_array(img, dim_ordering=K.image_dim_ordering()):
    if dim_ordering not in ['th', 'tf']:
        raise Exception('Unknown dim_ordering: ', dim_ordering)
    # image has dim_ordering (height, width, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise Exception('Unsupported image shape: ', x.shape)
    return x


def load_img(path, target_mode=None, target_size=None):
    from PIL import Image
    img = Image.open(path)
    if target_mode:
        img = img.convert(target_mode)
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img

def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and re.match('([\w]+\.(?:' + ext + '))', f)]

def pil_image_reader(filepath, target_mode=None, target_size=None, dim_ordering=K.image_dim_ordering(), **kwargs):
    img = load_img(filepath, target_mode=target_mode, target_size=target_size)
    return img_to_array(img, dim_ordering=dim_ordering)

def standardize(x,
                dim_ordering='th',
                rescale=False,
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                mean=None, std=None,
                samplewise_std_normalization=False,
                zca_whitening=False, principal_components=None,
                featurewise_standardize_axis=None,
                samplewise_standardize_axis=None,
                fitting=False,
                verbose=0,
                config={},
                **kwargs):
    '''

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        featurewise_standardize_axis: axis along which to perform feature-wise center and std normalization.
        samplewise_standardize_axis: axis along which to to perform sample-wise center and std normalization.
        zca_whitening: apply ZCA whitening.

    '''
    if fitting:
        if '_X' in config:
            # add data to _X array
            config['_X'][config['_iX']] = x
            config['_iX'] +=1
            # if verbose and config.has_key('_fit_progressbar'):
                # config['_fit_progressbar'].update(config['_iX'], force=(config['_iX']==fitting))

            # the array (_X) is ready to fit
            if config['_iX'] >= fitting:
                X = config['_X'].astype('float32')
                del config['_X']
                del config['_iX']
                if featurewise_center or featurewise_std_normalization:
                    featurewise_standardize_axis = featurewise_standardize_axis or 0
                    if type(featurewise_standardize_axis) is int:
                        featurewise_standardize_axis = (featurewise_standardize_axis, )
                    assert 0 in featurewise_standardize_axis, 'feature-wise standardize axis should include 0'

                if featurewise_center:
                    mean = np.mean(X, axis=featurewise_standardize_axis, keepdims=True)
                    config['mean'] = np.squeeze(mean, axis=0)
                    X -= mean

                if featurewise_std_normalization:
                    std = np.std(X, axis=featurewise_standardize_axis, keepdims=True)
                    config['std'] = np.squeeze(std, axis=0)
                    X /= (std + 1e-7)

                if zca_whitening:
                    flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
                    sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
                    U, S, V = linalg.svd(sigma)
                    config['principal_components'] = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)
                if verbose:
                    del config['_fit_progressbar']
        else:
            # start a new fitting, fitting = total sample number
            config['_X'] = np.zeros((fitting,)+x.shape)
            config['_iX'] = 0
            config['_X'][config['_iX']] = x
            config['_iX'] +=1
            # if verbose:
                # config['_fit_progressbar'] = Progbar(target=fitting, verbose=verbose)
        return x

    if rescale:
        x *= rescale

    # x is a single image, so it doesn't have image number at index 0
    if dim_ordering == 'th':
        channel_index = 0
    if dim_ordering == 'tf':
        channel_index = 2

    samplewise_standardize_axis = samplewise_standardize_axis or channel_index
    if type(samplewise_standardize_axis) is int:
        samplewise_standardize_axis = (samplewise_standardize_axis, )

    if samplewise_center:
        x -= np.mean(x, axis=samplewise_standardize_axis, keepdims=True)
    if samplewise_std_normalization:
        x /= (np.std(x, axis=samplewise_standardize_axis, keepdims=True) + 1e-7)

    if verbose:
        if (featurewise_center and mean is None) or (featurewise_std_normalization and std is None) or (zca_whitening and principal_components is None):
            print('WARNING: feature-wise standardization and zca whitening will be disabled, please run "fit" first.')

    if featurewise_center:
        if mean is not None:
            x -= mean
    if featurewise_std_normalization:
        if std is not None:
            x /= (std + 1e-7)

    if zca_whitening:
        if principal_components is not None:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
    return x

def center_crop(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[0]//2, x.shape[1]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[centerw-halfw:centerw+halfw,centerh-halfh:centerh+halfh, :]

def random_crop(x, random_crop_size, sync_seed=None, rng=None, **kwargs):
    # np.random.seed(sync_seed)
    w, h = x.shape[0], x.shape[1]
    rangew = (w - random_crop_size[0]) // 2
    rangeh = (h - random_crop_size[1]) // 2
    #print('w: {}, h: {}, rangew: {}, rangeh: {}'.format(w, h, rangew, rangeh))
    offsetw = 0 if rangew == 0 else rng.randint(rangew)
    offseth = 0 if rangeh == 0 else rng.randint(rangeh)
    return x[offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1], :]

from keras.applications.inception_v3 import preprocess_input as pp

def preprocess_input(x, rng=None, **kwargs):
    return pp(x)

def random_transform(x,
                     dim_ordering='tf',
                     rotation_range=0.,
                     width_shift_range=0.,
                     height_shift_range=0.,
                     shear_range=0.,
                     zoom_range=0.,
                     channel_shift_range=0.,
                     fill_mode='nearest',
                     cval=0.,
                     horizontal_flip=False,
                     vertical_flip=False,
                     rescale=None,
                     sync_seed=None,
                     rng=None,
                     **kwargs):
    '''

    # Arguments
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided (before applying
            any other transformation).
    '''
    # rng.seed(sync_seed)

    x = x.astype('float32')
    # x is a single image, so it doesn't have image number at index 0
    if dim_ordering == 'th':
        img_channel_index = 0
        img_row_index = 1
        img_col_index = 2
    if dim_ordering == 'tf':
        img_channel_index = 2
        img_row_index = 0
        img_col_index = 1
    # use composition of homographies to generate final transform that needs to be applied
    if rotation_range:
        theta = np.pi / 180 * rng.uniform(-rotation_range, rotation_range)
    else:
        theta = 0
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    if height_shift_range:
        tx = rng.uniform(-height_shift_range, height_shift_range) * x.shape[img_row_index]
    else:
        tx = 0

    if width_shift_range:
        ty = rng.uniform(-width_shift_range, width_shift_range) * x.shape[img_col_index]
    else:
        ty = 0

    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    if shear_range:
        shear = rng.uniform(-shear_range, shear_range)
    else:
        shear = 0
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    if np.isscalar(zoom_range):
        zoom_range = [1 - zoom_range, 1 + zoom_range]
    elif len(zoom_range) == 2:
        zoom_range = [zoom_range[0], zoom_range[1]]
    else:
        raise Exception('zoom_range should be a float or '
                        'a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = rng.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

    h, w = x.shape[img_row_index], x.shape[img_col_index]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    x = apply_transform(x, transform_matrix, img_channel_index,
                        fill_mode=fill_mode, cval=cval)
    if channel_shift_range != 0:
        x = random_channel_shift(x, channel_shift_range, img_channel_index, rng=rng)

    if horizontal_flip:
        if rng.rand() < 0.5:
            x = flip_axis(x, img_col_index)

    if vertical_flip:
        if rng.rand() < 0.5:
            x = flip_axis(x, img_row_index)

    # TODO:
    # barrel/fisheye

    #rng.seed()
    return x

class ImageDataGenerator(object):
    '''Generate minibatches with
    real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        featurewise_standardize_axis: axis along which to perform feature-wise center and std normalization.
        samplewise_standardize_axis: axis along which to to perform sample-wise center and std normalization.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided (before applying
            any other transformation).
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
        seed: random seed for reproducible pipeline processing. If not None, it will also be used by `flow` or
            `flow_from_directory` to generate the shuffle index in case of no seed is set.
    '''
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 featurewise_standardize_axis=None,
                 samplewise_standardize_axis=None,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 dim_ordering=K.image_dim_ordering(),
                 seed=None,
                 verbose=1):
        self.config = copy.deepcopy(locals())
        self.config['config'] = self.config
        self.config['mean'] = None
        self.config['std'] = None
        self.config['principal_components'] = None
        self.config['rescale'] = rescale

        if dim_ordering not in {'tf', 'th'}:
            raise Exception('dim_ordering should be "tf" (channel after row and '
                            'column) or "th" (channel before row and column). '
                            'Received arg: ', dim_ordering)

        self.__sync_seed = self.config['seed'] or np.random.randint(0, 4294967295)

        self.default_pipeline = []
        self.default_pipeline.append(random_transform)
        self.default_pipeline.append(standardize)
        self.set_pipeline(self.default_pipeline)

        self.__fitting = False
        # self.fit_lock = threading.Lock()

    @property
    def sync_seed(self):
        return self.__sync_seed

    @property
    def fitting(self):
        return self.__fitting

    @property
    def pipeline(self):
        return self.__pipeline

    def sync(self, image_data_generator):
        self.__sync_seed = image_data_generator.sync_seed
        return (self, image_data_generator)

    def set_pipeline(self, p):
        if p is None:
            self.__pipeline = self.default_pipeline
        elif type(p) is list:
            self.__pipeline = p
        else:
            raise Exception('invalid pipeline.')

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_mode=None, save_format='jpeg',
             pool=None):
        return NumpyArrayIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.config['dim_ordering'],
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_mode=save_mode, save_format=save_format,
            pool=pool)

    def flow_from_directory(self, directory,
                            color_mode=None, target_size=None,
                            image_reader='pil', reader_config={'target_mode':'RGB', 'target_size':(256,256)},
                            read_formats={'png','jpg','jpeg','bmp'},
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='',
                            save_mode=None, save_format='jpeg'):
        return DirectoryIterator(
            directory, self,
            color_mode=color_mode, target_size=target_size,
            image_reader=image_reader, reader_config=reader_config,
            read_formats=read_formats,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.config['dim_ordering'],
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_mode=save_mode, save_format=save_format)

    def process(self, x, rng):
        # get next sync_seed
        # np.random.seed(self.__sync_seed)
        #np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        #self.__sync_seed = np.random.randint(0, 4294967295)
        # __sync_seed = rng.randint(0, 4294967295)
        # # print(self.__sync_seed)
        # self.config['fitting'] = self.__fitting
        # try:
        #     del self.config['sync_seed']
        # except:
        #     pass
        #self.config['sync_seed'] = self.__sync_seed
        for p in self.__pipeline:
            x = p(x, rng=rng, **self.config)
        return x

    def fit_generator(self, generator, nb_iter):
        '''Fit a generator

        # Arguments
            generator: Iterator, generate data for fitting.
            nb_iter: Int, number of iteration to fit.
        '''
        # with self.fit_lock:
        #     try:
        #         self.__fitting = nb_iter*generator.batch_size
        #         for i in range(nb_iter):
        #             next(generator)
        #     finally:
        #         self.__fitting = False

    def fit(self, X, rounds=1):
        '''Fit the pipeline on a numpy array

        # Arguments
            X: Numpy array, the data to fit on.
            rounds: how many rounds of fit to do over the data
        '''
        X = np.copy(X)
        # with self.fit_lock:
        #     try:
        #         self.__fitting = rounds*X.shape[0]
        #         for r in range(rounds):
        #             for i in range(X.shape[0]):
        #                 self.process(X[i])
        #     finally:
        #         self.__fitting = False

class Iterator(object):

    def __init__(self, N, batch_size, shuffle, seed):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if self.batch_index == 0:
                self.index_array = np.arange(N)
                if shuffle:
                    if seed is not None:
                        np.random.seed(seed + self.total_batches_seen)
                    self.index_array = np.random.permutation(N)
                    if seed is not None:
                        np.random.seed()

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (self.index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __add__(self, it):
        assert self.N == it.N
        assert self.batch_size == it.batch_size
        assert self.shuffle == it.shuffle
        seed = self.seed or np.random.randint(0, 4294967295)
        it.total_batches_seen = self.total_batches_seen
        self.index_generator = self._flow_index(self.N, self.batch_size, self.shuffle, seed)
        it.index_generator = it._flow_index(it.N, it.batch_size, it.shuffle, seed)
        if (sys.version_info > (3, 0)):
            iter_zip = zip
        else:
            from itertools import izip
            iter_zip = izip
        return iter_zip(self, it)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

def process_image_worker(tup):
    process, img, rng = tup
    ret = process(img, rng)
    return ret

class NumpyArrayIterator(Iterator):

    def __init__(self, X, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering=K.image_dim_ordering(),
                 save_to_dir=None, save_prefix='',
                 save_mode=None, save_format='jpeg',
                 pool=None):
        if y is not None and len(X) != len(y):
            raise Exception('X (images tensor) and y (labels) '
                            'should have the same length. '
                            'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
        self.X = X
        self.y = y
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_mode = save_mode
        self.save_format = save_format
        seed = seed or image_data_generator.config['seed']
        self.pool = pool
        self.rngs = [np.random.RandomState(seed + i) for i in range(batch_size)]
        super(NumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def close(self):
        pass
        # print('closing pool!')
        # self.pool.close()
        # self.pool.join()
        # self.pool.terminate()
        # self.pool = None
        # print('closed pool!')

    def __add__(self, it):
        if isinstance(it, NumpyArrayIterator):
            assert self.X.shape[0] == it.X.shape[0]
        if isinstance(it, DirectoryIterator):
            assert self.X.shape[0] == it.nb_sample
        it.image_data_generator.sync(self.image_data_generator)
        return super(NumpyArrayIterator, self).__add__(it)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        result = self.pool.map(process_image_worker, ((self.image_data_generator.process, self.X[j], self.rngs[i%self.batch_size]) for i, j in enumerate(index_array)))
        batch_x = np.array(result)

        for i, rng in enumerate(self.rngs):
            new_seed = rng.randint(0, 4294967295)
            self.rngs[i] = np.random.RandomState(new_seed)

        # for i, j in enumerate(index_array):
        #     # print(i, j)
        #     x = self.X[j]
        #     x = self.image_data_generator.process(x)
        #     if i == 0:
        #         batch_x = np.zeros((current_batch_size,) + x.shape)
        #         # print(batch_x.shape)
        #     batch_x[i] = x

        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, mode=self.save_mode, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y


class DirectoryIterator(Iterator):

    def __init__(self, directory, image_data_generator,
                 color_mode=None, target_size=None,
                 image_reader="pil", read_formats={'png','jpg','jpeg','bmp'},
                 reader_config={'target_mode': 'RGB', 'target_size':None},
                 dim_ordering=K.image_dim_ordering,
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='',
                 save_mode=None, save_format='jpeg'):
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.image_reader = image_reader
        if self.image_reader == 'pil':
            self.image_reader = pil_image_reader
        self.reader_config = reader_config
        # TODO: move color_mode and target_size to reader_config
        if color_mode == 'rgb':
            self.reader_config['target_mode'] = 'RGB'
        elif color_mode == 'grayscale':
            self.reader_config['target_mode'] = 'L'

        if target_size:
            self.reader_config['target_size'] = target_size

        self.dim_ordering = dim_ordering
        self.reader_config['dim_ordering'] = dim_ordering
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_mode = save_mode
        self.save_format = save_format

        seed = seed or image_data_generator.config['seed']

        # first, count the number of samples and classes
        self.nb_sample = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        # if no class is found, add '' for scanning the root folder
        if class_mode is None and len(classes) == 0:
            classes.append('')
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for fname in os.listdir(subpath):
                is_valid = False
                for extension in read_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    self.nb_sample += 1
        print('Found %d images belonging to %d classes.' % (self.nb_sample, self.nb_class))

        # second, build an index of the images in the different class subfolders
        self.filenames = []
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        i = 0
        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for fname in os.listdir(subpath):
                is_valid = False
                for extension in read_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    self.classes[i] = self.class_indices[subdir]
                    self.filenames.append(os.path.join(subdir, fname))
                    i += 1

        assert len(self.filenames)>0, 'No valid file is found in the target directory.'
        self.reader_config['class_mode'] = self.class_mode
        self.reader_config['classes'] = self.classes
        self.reader_config['filenames'] = self.filenames
        self.reader_config['directory'] = self.directory
        self.reader_config['nb_sample'] = self.nb_sample
        self.reader_config['seed'] = seed
        self.reader_config['sync_seed'] = self.image_data_generator.sync_seed
        super(DirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)
        if inspect.isgeneratorfunction(self.image_reader):
            self._reader_generator_mode = True
            self._reader_generator = []
            # set index batch_size to 1
            self.index_generator = self._flow_index(self.N, 1 , self.shuffle, seed)
        else:
            self._reader_generator_mode = False

    def __add__(self, it):
        if isinstance(it, DirectoryIterator):
            assert self.nb_sample == it.nb_sample
            assert len(self.filenames) == len(it.filenames)
            assert np.alltrue(self.classes == it.classes)
            assert self.image_reader == it.image_reader
            if inspect.isgeneratorfunction(self.image_reader):
                self._reader_generator = []
                it._reader_generator = []
        if isinstance(it, NumpyArrayIterator):
            assert self.nb_sample == self.X.shape[0]
        it.image_data_generator.sync(self.image_data_generator)
        return super(DirectoryIterator, self).__add__(it)

    def next(self):
        self.reader_config['sync_seed'] = self.image_data_generator.sync_seed
        if self._reader_generator_mode:
            sampleCount = 0
            batch_x = None
            _new_generator_flag = False
            while sampleCount<self.batch_size:
                for x in self._reader_generator:
                    _new_generator_flag = False
                    if x.ndim == 2:
                        x = np.expand_dims(x, axis=0)
                    x = self.image_data_generator.process(x)
                    self.reader_config['sync_seed'] = self.image_data_generator.sync_seed
                    if sampleCount == 0:
                        batch_x = np.zeros((self.batch_size,) + x.shape)
                    batch_x[sampleCount] = x
                    sampleCount +=1
                    if sampleCount >= self.batch_size:
                        break
                if sampleCount >= self.batch_size or _new_generator_flag:
                    break
                with self.lock:
                    index_array, _, _ = next(self.index_generator)
                fname = self.filenames[index_array[0]]
                self._reader_generator = self.image_reader(os.path.join(self.directory, fname), **self.reader_config)
                assert isinstance(self._reader_generator, types.GeneratorType)
                _new_generator_flag = True
        else:
            with self.lock:
                index_array, current_index, current_batch_size = next(self.index_generator)
            # The transformation of images is not under thread lock so it can be done in parallel
            batch_x = None
            # build batch of image data
            for i, j in enumerate(index_array):
                fname = self.filenames[j]
                x = self.image_reader(os.path.join(self.directory, fname), **self.reader_config)
                if x.ndim == 2:
                    x = np.expand_dims(x, axis=0)
                x = self.image_data_generator.process(x)
                if i == 0:
                    batch_x = np.zeros((current_batch_size,) + x.shape)
                batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, mode=self.save_mode, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y
