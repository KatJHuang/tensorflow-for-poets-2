import argparse
import hashlib
import os
import re
from pathlib import Path

import tensorflow as tf
from tensorflow.python.platform import gfile

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def write_images_to_tfrecords(image_dir, output_dir):
    if not gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    sub_dirs = [
        os.path.join(image_dir, item)
        for item in gfile.ListDirectory(image_dir)]
    sub_dirs = sorted(item for item in sub_dirs
                      if gfile.IsDirectory(item))

    how_many = 0

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = str(Path(output_dir) / Path("images.tfrecords"))
    writer = tf.python_io.TFRecordWriter(filename)

    for sub_dir in sub_dirs:
        print('Processing images in folder {}'.format(sub_dir))
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        if len(file_list) < 20:
            tf.logging.warning(
                'WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning(
                'WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # We want to ignore anything after '_nohash_' in the file name when
            # deciding which set to put an image in, the data set creator has a way of
            # grouping photos that are close variations of each other. For example
            # this is used in the plant disease data set to group multiple pictures of
            # the same leaf.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()

            image_data = gfile.FastGFile(file_name, 'rb').read()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_name.encode('utf-8')])),
                'hash_name_hashed':
                    tf.train.Feature(bytes_list=tf.train.BytesList(value=[hash_name_hashed.encode('utf-8')])),
                'image_data':
                    tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data]))
            }))
            writer.write(example.SerializeToString())

            how_many += 1
            if how_many % 100 == 0:
                print('Written {} images to tfrecords.'.format(how_many))

    writer.close()

def main():
    parser = argparse.ArgumentParser(description='Convert data for model training into TF Records.')
    parser.add_argument('input',
                        help="""
                        Directory containing all training data with the following structure:
                        data\
                            label1\
                                1_img1.jpg
                                ...
                            label2\
                                2_img1.jpg
                                ...
                            ...
                        """)
    parser.add_argument('output',
                        default='serialized_image_data/',
                        help='Directory to save output imgs to.')

    args = parser.parse_args()
    write_images_to_tfrecords(args.input, args.output)


if __name__ == '__main__':
    main()
