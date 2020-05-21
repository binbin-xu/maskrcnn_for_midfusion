import os
import sys
import numpy as np
import cv2
import csv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo


def getVideoFile():
    # for arg in sys.argv[1:]:
    # for arg in sys.argv[1]:
    #     video_file = arg
    video_file = sys.argv[1]

    return video_file


if __name__ == '__main__':
    """
        test everything
    """
    # ROOT_DIR = os.path.abspath("../")
    # sys.path.append(ROOT_DIR)  # To find local version of the library
    # MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # video_file = "/media/binbin/data/dataset/dynamic/rot_chair/rgb.mp4"
    video_file = getVideoFile()

    # Import Mask RCNN
    # cnn_path = '/home/binbin/code/Tool/tensorflow/tensorpackZoo/COCO-ResNet50-MaskRCNN.npz'
    config_file = "../configs/caffe2/e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml"
    cfg.merge_from_file(config_file)
    coco_demo = COCODemo(
        cfg,
        min_image_size=100,
        confidence_threshold=0.5,
    )

    video_path = os.path.dirname(video_file)

    capture = cv2.VideoCapture(video_file)
    frame_count = 0

    # build folders
    mask_rcnn_path = os.path.join(video_path, "mask_RCNN")
    if not os.path.exists(mask_rcnn_path):
        os.makedirs(mask_rcnn_path)

    visual_path = os.path.join(mask_rcnn_path, "visualization")
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)

    class_path = os.path.join(mask_rcnn_path, "class_id")
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    mask_path = os.path.join(mask_rcnn_path, "mask")
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    box_path = os.path.join(mask_rcnn_path, "box")
    if not os.path.exists(box_path):
        os.makedirs(box_path)

    prob_path = os.path.join(mask_rcnn_path, "prob")
    if not os.path.exists(prob_path):
        os.makedirs(prob_path)

    all_prob_path = os.path.join(mask_rcnn_path, "all_prob")
    if not os.path.exists(all_prob_path):
        os.makedirs(all_prob_path)

    csv_path = os.path.join(mask_rcnn_path, "maskrcnn.csv")

    out = csv.writer(open(csv_path, "w", newline=''), delimiter=',', quoting=csv.QUOTE_ALL)
    data = ["#visualisation", "mask", "class-id", "box", "prob", "all_prob"]
    out.writerow(data)

    while (capture.isOpened()):
        ret, frame = capture.read()
        # Bail out when the video file ends
        if not ret:
            break

        print('frame_count :{0}'.format(frame_count))
        all_probs, box, class_id, mask, prob, visualization = coco_demo.output_predictions(frame)
        # plt.imshow(visualization[:, :, [2, 1, 0]])
        # plt.show()
        # print('Predicted')

        # do not output bg probablity: legacy issue in mid-fusion
        all_probs = all_probs[:, 1:]

        # save visualization
        name = '{0:04d}.png'.format(frame_count)
        visual_name = os.path.join(visual_path, name)
        cv2.imwrite(visual_name, visualization)
        print('writing to file:{0}'.format(visual_name))

        # save mask
        name = '{0:04d}'.format(frame_count)
        mask_name = os.path.join(mask_path, name)
        np.save(mask_name, mask)
        print('writing to file:{0}'.format(mask_name))

        # save class id
        class_name = os.path.join(class_path, name)
        np.save(class_name, class_id)
        print('writing to file:{0}'.format(class_name))

        # save box
        box_name = os.path.join(box_path, name)
        np.save(box_name, box)
        print('writing to file:{0}'.format(box_name))

        # probability #N
        prob_name = os.path.join(prob_path, name)
        np.save(prob_name, prob)
        print('writing to file:{0}'.format(prob_name))

        # all probability #N * #class
        all_prob_name = os.path.join(all_prob_path, name)
        np.save(all_prob_name, all_probs)
        print('writing to file:{0}'.format(all_prob_name))

        # save associsations
        data = [visual_name, mask_name, class_name, box_name, prob_name, all_prob_name]
        out.writerow(data)

        frame_count += 1

    capture.release()