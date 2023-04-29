import os
import xml.etree.ElementTree as ET
import itertools

import numpy as np


def getTruth():
    data_path = r'C:\Users\mrcry\Documents\data\voc\VOCtest-2007'

    with open('pred.txt', 'r') as f:
        lines = f.readlines()
        image_ids = [line.strip().split()[0] for line in lines]

    with open('truth.txt', 'a') as t:
        for image_id in image_ids[300:500]:
            label_path = os.path.join(data_path, 'Annotations-all', image_id + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            t.write(image_id + ' ')
            for obj in objects:
                bbox = obj.find('bndbox')
                xmin = bbox.find('xmin').text.strip()
                ymin = bbox.find('ymin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymax = bbox.find('ymax').text.strip()

                t.write(xmin + ' ' + xmax + ' ' + ymin + ' ' + ymax + ' ')
                print(xmin, xmax, ymin, ymax)
            t.write('\n')


def getParameters():
    with open(r'C:\Users\mrcry\Documents\yolo-v3-36\utils\pred.txt', 'r') as f1:
        lines = f1.readlines()
        pred_images = [line.strip().split() for line in lines]

    with open(r'C:\Users\mrcry\Documents\yolo-v3-36\utils\truth.txt', 'r') as f2:
        lines = f2.readlines()
        true_images = [line.strip().split() for line in lines]

    M = []
    for (p, t) in zip(pred_images, true_images):
        if len(p) < 4:
            continue
        elif p[0] == t[0]:
            obj_num = (len(p) - 1) // 4
            for i in range(0, obj_num, 4):
                pxmin, pxmax, pymin, pymax = float(p[i + 1]), float(p[i + 2]), float(p[i + 3]), float(p[i + 4])
                txmin, txmax, tymin, tymax = int(t[i + 1]), int(t[i + 2]), int(t[i + 3]), int(t[i + 4])
                px = (pxmin + pxmax) // 2
                py = (pymin + pymax) // 2
                pw = pxmax - pxmin
                ph = pymax - pymin

                tx = (txmin + txmax) // 2
                ty = (tymin + tymax) // 2
                tw = txmax - txmin
                th = tymax - tymin

                M.append(
                    [px, py, pw, ph, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -tx * px, -tx * py, -tx * pw,
                     -tx * ph,
                     -tx])
                M.append(
                    [0, 0, 0, 0, 0, px, py, pw, ph, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -ty * px, -ty * py, -ty * pw,
                     -ty * ph,
                     -ty])
                M.append(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, px, py, pw, ph, 1, 0, 0, 0, 0, 0, -tw * px, -tw * py, -tw * pw,
                     -tw * ph,
                     -tw])
                M.append(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, px, py, pw, ph, 1, -th * px, -th * py, -th * pw,
                     -th * ph,
                     -th])

    # compute At x A
    A = np.array(M)
    return gradient_descent(A).reshape(5,5)
    # return np.linalg.solve(A_, np.zeros(A_.shape))

    # eigenvalues, eigenvectors = np.linalg.eig(A_)
    # a = eigenvectors[:, 24]
    # A = a.reshape(5, 5)
    # return A

def gradient_descent(A, interation = 1000, learning_rate = 0.0001):
    f = open('gradient_result.txt','w')
    p_0 = np.random.randint(low=10, size=(25, 1))
    f.write("initial point: ")
    f.write(str(p_0)+'\n')
    for i in range(interation):
        dp = np.dot(A.T, A).dot(p_0)
        dp_hat = dp / np.linalg.norm(dp)
        f.write("gradient: "+str(dp_hat)+'\n')
        p = p_0 - learning_rate * dp_hat
        f.write("object value:"+ str((np.dot(p.T,A.T).dot(A).dot(p))) + '\n')
        if abs(p_0 - p).all() < 0.0001:
            return p
        p_0 = p
    f.close()
    return p_0



def calibrate(xmin, xmax, ymin, ymax):
    A = getParameters()
    print(A)
    x = (xmin + xmax) // 2
    y = (ymin + ymax) // 2
    w = xmax - xmin
    h = ymax - ymin
    X = np.array([x, y, w, h, 1])
    B = np.dot(A, X)
    cxmin = B[0] - B[2] / 2
    cxmax = B[0] + B[2] / 2
    cymin = B[1] - B[3] / 2
    cymax = B[1] + B[3] / 2
    return cxmin, cxmax, cymin, cymax


if __name__ == "__main__":
    getParameters()
