import glob

import numpy as np
from matplotlib import pyplot as plt
from scipy import misc

import funcs
from MyViola import MyViolaClassifier


def find_face(img, shape, mv):
    res_i = (0, 0)
    res_j = (0, 0)
    res_scl = 1
    scl = 1
    max_ = 0
    m, n = shape
    while img.shape[0] > 400 or img.shape[1] > 400:
        img = misc.imresize(img, 0.7)
        scl *= 0.7
    while img.shape[0] > m and img.shape[1] > n:
        mv.change_image(img)
        x, y = img.shape[:2]
        if x < m or y < n:
            continue
        for i, j in funcs.iter_shape((x, y), shape, 4):
            val = mv.valuefy((i, j))
            if val > max_:
                max_ = val
                res_i, res_j = i, j
                res_scl = scl
        img = misc.imresize(img, 0.8)
        scl *= 0.8
    return (int(res_i[0] / res_scl), int(res_i[1] / res_scl)), (int(res_j[0] / res_scl), int(res_j[1] / res_scl))


def get_sub_pics_with_size(imgs, shape):
    scales = np.arange(.2, 1, .2)
    m, n = shape
    for img in imgs:
        yield misc.imresize(img, shape)
        while img.shape[0] > 200:
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
    #mv.add_examples(get_sub_pics_with_size(nofaces, (70, 70)), -1)
    mv.add_examples((misc.imresize(nf, (70, 70)) for nf in nofaces), -1)
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
    m, n = shape
    base_m, base_n = 600, 600
    scl = 1
    while img.shape[0] > base_m or img.shape[1] > base_n:
        img = misc.imresize(img, 0.8)
        scl *= 0.8
    while img.shape[0] > m and img.shape[1] > n:
        mv.change_image(img)
        x, y = img.shape[:2]
        for i, j in funcs.iter_shape((x, y), shape, 4):
            val = mv.valuefy((i, j))
            if val > 0:
                res_i = (int(i[0] / scl), int(i[1] / scl))
                res_j = (int(j[0] / scl), int(j[1] / scl))
                yield ((res_i, res_j), val)
        img = misc.imresize(img, 0.8)
        scl *= 0.8


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


def find_with_rotate(img, shape, mv):
    angle = 22
    angle_rad = angle / 180 * np.pi
    normal = get_all_faces_rects(img, shape, mv)
    im_right = misc.imrotate(img, angle)
    right = get_all_faces_rects(im_right, shape, mv)
    im_left = misc.imrotate(img, -angle)
    left = get_all_faces_rects(im_left, shape, mv)
    rot_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    for r in normal:
        yield r
    for r in left:
        m, n = im_left.shape[:2]
        x, y = r
        t = rot_mat.dot([x[0] - m / 2, y[0] - n / 2])[0] + m / 2  # top
        b = rot_mat.dot([x[1] - m / 2, y[1] - n / 2])[0] + m / 2  # bottom
        l = rot_mat.dot([x[0] - m / 2, y[1] - n / 2])[1] + n / 2  # left
        r = rot_mat.dot([x[1] - m / 2, y[0] - n / 2])[1] + n / 2  # right
        yield ((t, b), (l, r))
    rot_mat = rot_mat.T
    for r in right:
        m, n = im_right.shape[:2]
        x, y = r
        l = rot_mat.dot([x[0] - m / 2, y[0] - n / 2])[1] + n / 2  # left
        r = rot_mat.dot([x[1] - m / 2, y[1] - n / 2])[1] + n / 2  # right
        t = rot_mat.dot([x[0] - m / 2, y[1] - n / 2])[0] + m / 2  # top
        b = rot_mat.dot([x[1] - m / 2, y[0] - n / 2])[0] + m / 2  # bottom
        yield ((t, b), (l, r))
