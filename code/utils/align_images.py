import os
import sys
import bz2
import argparse

from utils.face_alignment import image_align
from utils.landmarks_detector import LandmarksDetector
import multiprocessing

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

def align_images(image1, image2):
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """

    output_size = 1024
    landmarks_detector = LandmarksDetector()
    images = [image1, image2]

    print(images)

    for img_name in images:
        print('Aligning %s ...' % img_name)
        try:
            raw_img_path = img_name
            print('Loading image...', raw_img_path)
            fn = face_img_name = '%s_aligned.png' % (os.path.splitext(img_name)[0])
            print('Getting landmarks...', fn)
            if os.path.isfile(fn):
                continue
            for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                try:
                    face_img_name = '%s_aligned.png' % (os.path.splitext(img_name)[0])
                    print('Starting face alignment...', face_img_name)
                    aligned_face_path = face_img_name
                    print(raw_img_path, aligned_face_path, face_landmarks, output_size)
                    image_align(raw_img_path, aligned_face_path, face_landmarks, output_size=output_size)
                    print('Wrote result %s' % aligned_face_path)
                except:
                    print("Exception in face alignment!")
        except:
            print("Exception in landmark detection!")