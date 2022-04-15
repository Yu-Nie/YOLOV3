import cv2
import os
from PIL import Image
import xml.etree.ElementTree as ET
import xml.dom.minidom

data_path = r"C:\Users\mrcry\Documents\data\voc"

'''
this function create new trainval.txt for VOC2007, VOC2012, new test.txt for VOCtest-2007
these are image ids that have corresponding segmentationObject
'''

def seg_data():
    years = ['VOC2007', 'VOC2012', 'VOCtest-2007']
    id_list_2007 = []
    id_list_2012 = []
    id_list_test = []
    for year in years:
        seg = os.path.join(data_path, year, 'SegmentationObject')
        trainval = os.path.join(data_path, year, 'ImageSets', 'Main', 'trainval.txt')
        trainval_new = os.path.join(data_path, year, 'ImageSets', 'Main', 'trainval_new.txt')
        if year == 'VOC2007':
            for pic in os.listdir(seg):
                id_list_2007.append(pic.split('.')[0])
            with open(trainval, 'r') as rd, open(trainval_new, 'w') as wt:
                for index, line in enumerate(rd):
                    if line.strip() in id_list_2007:
                        wt.write(line)
        elif year == 'VOC2012':
            for pic in os.listdir(seg):
                id_list_2012.append(pic.split('.')[0])
            with open(trainval, 'r') as rd, open(trainval_new, 'w') as wt:
                for index, line in enumerate(rd):
                    if line.strip() in id_list_2012:
                        wt.write(line)
        else:
            for pic in os.listdir(seg):
                id_list_test.append(pic.split('.')[0])
            test_path = os.path.join(data_path, year, 'ImageSets', 'Main', 'test.txt')
            test_new = os.path.join(data_path, year, 'ImageSets', 'Main', 'test_new.txt')
            with open(test_path, 'r') as rd, open(test_new, 'w') as wt:
                for index, line in enumerate(rd):
                    if line.strip() in id_list_test:
                        wt.write(line)




def get_ratio(xmin, ymin, xmax, ymax, label, img_path):
    # stored as BGR
    color_label = [[0, 0, 128],
                   [0, 128, 0],
                   [0, 128, 128],
                   [128, 0, 0],
                   [128, 0, 128],
                   [128, 128, 0],
                   [128, 128, 128],
                   [0, 0, 64],
                   [0, 0, 192],
                   [0, 128, 64],
                   [0, 128, 192],
                   [128, 0, 64],
                   [128, 0, 192],
                   [128, 128, 64],
                   [128, 128, 192],
                   [0, 64, 0],
                   [0, 64, 128],
                   [0, 192, 0],
                   [0, 192, 128],
                   [128, 64, 0],
                   [128, 64, 128],
                   [128, 192, 0],
                   [128, 192, 128],
                   [0, 64, 64],
                   [0, 64, 192],
                   [0, 192, 64],
                   [0, 192, 192],
                   [128, 64, 64],
                   [128, 64, 192],
                   [128, 192, 64],
                   [128, 192, 192],
                   [64, 0, 0],
                   [64, 0, 128],
                   [64, 128, 0],
                   [64, 128, 128],
                   [192, 0, 0],
                   [192, 0, 128],
                   [192, 128, 0]]
    img = cv2.imread(img_path)
    ratio = []
    bt, gt, rt = color_label[label]
    xdiff = int((xmax - xmin) // 6)
    ydiff = int((ymax - ymin) // 6)
    if xdiff == 0 or ydiff == 0:
        ratio = [0 for _ in range(36)]
        return ratio
    for i in range(6):
        for j in range(6):
            segment = img[ymin + j * ydiff: ymin + (j + 1) * ydiff, xmin + i * xdiff:xmin + (i + 1) * xdiff, :]
            # cv2.imshow("img", segment)
            # cv2.waitKey(0)
            right, total = 0, 0
            list_of_pixels = segment.tolist()
            for point in list_of_pixels[0]:
                b, g, r = point
                if r == rt and g == gt and b == bt:
                    right += 1
                total += 1
            tmp = right / total if total != 0 else 0
            ratio.append(tmp)
    return ratio


'''
this function generate 36 ratio in each bounding box and add them to xml files
'''

def edit_annotaions():
    anno_path = os.path.join(data_path, 'VOC2012', 'Annotations-all')

    for f in os.listdir(anno_path):
        id = f.split('.')[0]
        image_xml_path = os.path.join(anno_path, "{}.xml".format(id))
        segment_path = os.path.join(data_path, 'VOC2012', 'SegmentationObject', "{}.png".format(id))
        annot = ET.parse(image_xml_path)
        label = 0
        # print(image_xml_path)
        for obj in annot.findall('object'):
            xmin, xmax, ymin, ymax = [obj.find('bndbox').find(tag).text for tag in
                                      ["xmin", "xmax", "ymin", "ymax"]]
            # print(xmin, xmax, ymin, ymax)
            # generate mask_ratio in xml files
            if os.path.isfile(segment_path):
                ratio = get_ratio(int(xmin), int(ymin), int(xmax), int(ymax), label, segment_path)
                mask_name = ET.Element('mask_ratio')
                mask_name.text = str(ratio)[1:-1]
                bbox = obj.find('bndbox')
                bbox.append(mask_name)
                # root = annot.getroot()
                # pretty_xml(root, '\t', '\n')
                new_path = image_xml_path.replace('Annotations-all', 'Annotations-36')
                annot.write(new_path)
                dom = xml.dom.minidom.parse(new_path)  # or xml.dom.minidom.parseString(xml_string)
                pretty_xml_as_string = dom.toprettyxml()
            label += 1


if __name__ == "__main__":
    # seg_data()
    edit_annotaions()
