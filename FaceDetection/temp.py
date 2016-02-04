import numpy as np
from scipy import ndimage, misc
from matplotlib import pyplot as plt
import glob
from MyViola import MyViolaClassifier
from Svm import Svm
import funcs


def find_face(img, shape, mv):
    res_i = (0, 0)
    res_j = (0, 0)
    res_scl = 1
    max_ = 0
    scales = np.arange(.2, .35, .025)
    m, n = shape
    for scl in scales:
        img_ = misc.imresize(img, scl)
        mv.change_image(img_)
        x, y = img_.shape[:2]
        if x < m or y < n:
            continue
        for i, j in funcs.iter_shape((x, y), shape, 4):
            val = mv.valuefy((i, j))
            if val > max_:
                max_ = val
                res_i, res_j = i, j
                res_scl = scl
    return (int(res_i[0] / res_scl), int(res_i[1] / res_scl)), (int(res_j[0] / res_scl), int(res_j[1] / res_scl))


def get_sub_pics_with_size(imgs, shape):
    scales = np.arange(.2, 1, .2)
    m, n = shape
    for img in imgs:
        yield misc.imresize(img, shape)
        while img.shape[0] > 800:
            img = misc.imresize(img, 0.5)
        for scl in scales:
            img_ = misc.imresize(img, scl)
            x, y = img_.shape[:2]
            if x < m or y < n:
                continue
            i = 0
            while i + m < x:
                j = 0
                while j + n < y:
                    yield img_[i:i+m, j:j+n]
                    j += n
                i += m


def temp():
    files = glob.glob('../faces/cropped/*.jpg')
    faces = (misc.imread(im) for im in files)
    mv = MyViolaClassifier()
    mv.add_examples(faces, 1)
    files = glob.glob('../faces/nofaces/*.jpg')
    nofaces = (misc.imread(im) for im in files)
    mv.add_examples(get_sub_pics_with_size(nofaces, (70, 70)), -1)
    mv.learn()
    mv.save('my_viola.pkl')
    files = glob.glob('../faces/*.jpg')
    for f in files:
        img = misc.imread(f)
        new_path = f.replace('/faces\\', '/faces\\new1\\')
        i, j = find_face(img, (70, 70), mv)
        i1, i2 = i
        j1, j2 = j
        new_img = img[i1:i2, j1:j2]
        try:
            misc.imsave(new_path, new_img)
        except ValueError:
            pass


def plot_image_faces(img, shape, mv):
    plot_im_with_rects(img, get_all_faces_rects(img, shape, mv))


def plot_im_with_rects(img, rect_list):
    img1 = img
    for rect in rect_list:
        img1 = funcs.implusrect(img1, rect[0], rect[1], (0, 255, 0))
    plt.imshow(img1)


def get_all_faces_rects(img, shape, mv):
    return [a[0] for a in filter_overlap_windows(get_all_windows(img, shape, mv))]


def get_all_windows(img, shape, mv):
    scales = np.arange(.2, .35, .02)
    m, n = shape
    for scl in scales:
        img_ = misc.imresize(img, scl)
        mv.change_image(img_)
        x, y = img_.shape[:2]
        if x < m or y < n:
            continue
        for i, j in funcs.iter_shape((x, y), shape, 4):
            val = mv.valuefy((i, j))
            if val > 0:
                res_i = (int(i[0] / scl), int(i[1] / scl))
                res_j = (int(j[0] / scl), int(j[1] / scl))
                yield ((res_i, res_j), val)


def is_pos_in_rect(pos, rect):
    x, y = pos
    (i1, i2), (j1, j2) = rect
    return i1 <= x <= i2 and j1 <= y <= j2


def mid_point(rect):
    (i1, i2), (j1, j2) = rect
    return int((i1 + i2) / 2), int((j1 + j2) / 2)


def are_overlap(window1, window2):
    return is_pos_in_rect(mid_point(window1), window2) or is_pos_in_rect(mid_point(window2), window1)


def filter_overlap_windows(windows):
    maxs = []
    for w in windows:
        w_waiting = True
        index = 0
        while index < len(maxs) and w_waiting:
            if are_overlap(w[0], maxs[index][0]):
                if w[1] > maxs[index][1]:
                    maxs[index] = w
                w_waiting = False
            index += 1
        if w_waiting:
            maxs.append(w)
    return maxs

