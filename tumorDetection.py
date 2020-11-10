import os, sys
import cv2
import numpy as np
import random
import glob
from PIL import Image
import os.path as osp
# E:\camelyon16\level2_1024\tumor_slide_mask
# E:\camelyon16\dataset_for_training\cam2017\level1\patient_004_node_4\\masks_1
allMaskPath ="E:\\camelyon16\\dataset_for_training\\10k_dataset\\images_level1_10k\\normal_tumor_mask\\"
allPatches = "E:\\camelyon16\\dataset_for_training\\level_1\\n\\"
maskDst = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\images_level1_10k\\normal_tumor_mask\\"

tumorDst = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\images_level1_10k\\normal\\"
# tumorDst_1 = "E:\\camelyon16\\level_2\\n_t_mask_1\\"
allMaskPathdirs = os.listdir(allMaskPath)
allPatchesdirs = os.listdir(tumorDst)
maskdstDir = os.listdir(maskDst)
images = glob.glob(allPatches + "*.png")

images.sort()
# maskdstDir_1 = os.listdir(tumorDst_1)

def is_sorta_black(arr, threshold= 0.1):
    tot = np.float(np.sum(arr))
    # print (tot / arr.size)
    # print (tot)
    # print (arr.size)

    # if (tot / arr.size > (threshold)|((tot/arr.size) == 1)):
    if (tot / arr.size > (threshold)):
    # if (tot / arr.size != 255):
    #    print ("is not black" )
       return True
    else:
       # print ("is kinda black")
       return False
def tottaly_white(arr, threshold= 1):
    tot = np.float(np.sum(arr))
    # print (tot / arr.size)
    # print (tot)
    # print (arr.size)

    # if (tot / arr.size > (threshold)|((tot/arr.size) == 1)):
    if (tot / arr.size == (threshold)):
    # if (tot / arr.size != 255):
    #    print ("is not black" )
       return True
    else:
       # print ("is kinda black")
       return False

# Detecting masks with tumor area
def tumor_mask_detection():
    for item in allMaskPathdirs:
        if os.path.isfile(allMaskPath + item):
            filenameMask, e = os.path.splitext(item)
            im = cv2.imread(allMaskPath + item,0)
            parts = filenameMask.split('_')
            # cv2.imwrite(maskDst + parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3] + ".png", image)
            # pix = im.load()
            # [x,y]= im.size  # Get the width and hight of the image for iterating over
            if(tottaly_white(im) == False):
                im = im/255
            # if (is_sorta_black(im)): # Get the RGBA Value of the a pixel of an image
            #     im = im *255
            cv2.imwrite(maskDst + parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3] + ".png", im)
            # cv2.imwrite(maskDst+item,im)
            # pix[x, y] = value  # Set the RGBA Value of the image (tuple)
            # im.save('alive_parrot.png')  # Save the modified pixels as .png

            # imResize = im.resize((256, 256), Image.ANTIALIAS)
            # imResize.convert('RGB').save(f+'.png', 'png', quality=80)

# Finding the same tumor image with the same mask name and copy to the specicif folder
def copy_tumor_with_same_mask_name():
    # searching by two for
    # You can use it with one 'for'. Searching with its name
    # random selection function is written with one
    for item in maskdstDir:
        for item2 in allPatchesdirs:

            filenameMask, e = os.path.splitext(item)
            fileNameTumor, e1 = os.path.splitext(item2)
            if (filenameMask == fileNameTumor+"_mask"):
                im = Image.open(allPatches + fileNameTumor+".png")
                im.save(tumorDst + item2)
def copy_mask_with_same_tumor_name():
    for item2 in allPatchesdirs:
        for item in allMaskPathdirs:

            filenameMask, e = os.path.splitext(item)
            fileNameTumor, e1 = os.path.splitext(item2)
            if (filenameMask == fileNameTumor+"_mask"):
                im = Image.open(allMaskPath + filenameMask+".png")
                im.save(tumorDst + item)
def copy_mask_with_same_tumor_name_2():
    for item in allPatchesdirs:
        filenameMask, e = os.path.splitext(item)
        parts = filenameMask.split('_')
        im = cv2.imread(allMaskPath + parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3] + "_mask"+".png", 1)
        # if (tottaly_white(im) == False):
        #     im = im / 255

        # fileNameTumor, e1 = os.path.splitext(item2)
        # im = Image.open(allMaskPath + filenameMask+"_mask.png")
        cv2.imwrite(maskDst + parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3] + ".png", im)
        # im.save(maskDst + item)
def create_empty_mask_for_patches():
    for item in allPatchesdirs:
        filenameMask, e = os.path.splitext(item)
        img = np.zeros([256, 256, 1], dtype=np.uint8)
        img.fill(0)  # or img[:] = 255
        # im = Image.open(allMaskPath + filenameMask+".png")
        cv2.imwrite(tumorDst + filenameMask+"_mask"+".png",img)
        # img.save(tumorDst_1 + filenameMask+"mask"+".png")
def convert_four_channels_to_three():
    for item in allPatchesdirs:
        if os.path.isfile(tumorDst + item):
            im = cv2.imread(tumorDst + item, 1)
            # rgb_binary = cv2.inRange(im, 100, 255)
            img = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
            cv2.imwrite(tumorDst + item, img)
            # pix = im.load()
            # [x,y]= im.size  # Get the width and hight of the image for iterating over
def resize():
    for item in allMaskPathdirs:
        filenameMask, e = os.path.splitext(item)
        image = Image.open(allMaskPath + filenameMask+".png")
        new_image = image.resize((1024, 1024))
        new_image = new_image / 255
        new_image.save(maskDst + filenameMask+".png")
def mask_division():
    # try:
    #     os.mkdir(maskDst+"/results")
    #     os.mkdir(maskDst+"/masks_1")
    # except OSError:
    #     print("Creation of the directory %s failed" % maskDst)
    # else:
    #     print("Successfully created the directory %s " % maskDst)
    for item in allMaskPathdirs:
        filenameMask, e = os.path.splitext(item)
        parts = filenameMask.split('_')
        image = cv2.imread(allMaskPath + filenameMask + ".png", 1)
        if (tottaly_white(image) == False):
            image = image / 255
        # image = image * 255
        # cv2.imwrite(maskDst + parts[0]+"_"+parts[1]+"_"+parts[2]+"_"+parts[3]+".png", image)
        cv2.imwrite(maskDst + filenameMask+".png", image)
        # cv2.imwrite(maskDst +filenameMask+".png", image)
def find_specific_image_from_folder():
    for item in allMaskPathdirs:
        filenameMask, e = os.path.splitext(item)
        parts = filenameMask.split('_')
        image = cv2.imread(allMaskPath + filenameMask+".png",1)
        cv2.imwrite(maskDst + parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3] + ".png", image)
        # cv2.imwrite(maskDst + filenameMask + ".png", image)

def random_selection_mask_img_pair():
    for i in range(1):
        key = random.choice(list(images))
        filename = osp.splitext(osp.basename(key))[0]
        im_mask = Image.open(allMaskPath + filename + "_mask.png")
        im = Image.open(allPatches + filename + ".png")
        im.save(tumorDst + filename + ".png")
        im_mask.save(maskDst + filename + ".png")
        print(key)
def is_sorta_tumor(arr, threshold=0.1):
    tot = np.float(np.sum(arr))
    print (tot)
    print(tot/arr.size )
    if tot/arr.size <(threshold):
       # print ("is not black" )
       return False
    else:
       # print ("is kinda black")
       return True

def get_confusion_matrix_intersection_mats(groundtruth, predicted):
    """ Returns dict of 4 boolean numpy arrays with True at TP, FP, FN, TN
    """

    confusion_matrix_arrs = {}

    groundtruth_inverse = np.logical_not(groundtruth)
    predicted_inverse = np.logical_not(predicted)

    confusion_matrix_arrs['tp'] = np.logical_and(groundtruth, predicted)
    tp = np.count_nonzero(confusion_matrix_arrs['tp'])
    confusion_matrix_arrs['tn'] = np.logical_and(groundtruth_inverse, predicted_inverse)
    tn = np.count_nonzero(confusion_matrix_arrs['tn'])
    confusion_matrix_arrs['fp'] = np.logical_and(groundtruth_inverse, predicted)
    fp = np.count_nonzero(confusion_matrix_arrs['fp'])
    confusion_matrix_arrs['fn'] = np.logical_and(groundtruth, predicted_inverse)
    fn = np.count_nonzero(confusion_matrix_arrs['fn'])
    if(is_sorta_tumor(groundtruth) & is_sorta_tumor(predicted)):
        dsc = (2 * tp) / (2 * tp + fp + fn)
        precision = tp / (tp + fn)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    return tp, tn, fp, fn, dsc, f1
def cofusion_matrix_two_folder():
    for item in allMaskPathdirs:
        filenameMask, e = os.path.splitext(item)
        gt = cv2.imread("E:\\camelyon16\\dataset_for_training\\level_2\\new_dataset\\test_unet\\mask\\" + filenameMask+".png", 0)
        pr = cv2.imread("E:\\camelyon16\\dataset_for_training\\level_2\\new_dataset\\test_unet\\output\\" + filenameMask+".png", 0)
        tp, tn, fp, fn, dsc, f1 = get_confusion_matrix_intersection_mats(gt, pr)
        print(tp)
        print(tn)
        print(fp)
        print(fn)
        print(dsc)
        print(f1)
def for_loop():
    for n in list(range(1, 10)) + list(range(10, 60, 10)):
        n = float(n)
        n = float(n/ 1000)
        print(n)
    # do something
    # for i in np.arange(0.0, 0.01,0.001) + np.arange(0.10,0.05,0.01):
    #     print(i)


if __name__ == '__main__':
    # resize()
    # mask_division()
    mask_division()
    # copy_mask_with_same_tumor_name_2()