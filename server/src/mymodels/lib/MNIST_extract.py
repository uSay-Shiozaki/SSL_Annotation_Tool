import os
import numpy as np
import matplotlib.pyplot as plt
import struct
from PIL import Image
PATH = '/mnt/media/irielab/HDD(3TB)/workspace/MNIST/raw'

trainImagesFile = open(os.path.join(PATH,'./train-images-idx3-ubyte'),'rb')
trainLabelsFile = open(os.path.join(PATH,'./train-labels-idx1-ubyte'),'rb')

f = trainImagesFile
l = trainLabelsFile

magic_number = f.read( 4 )
magic_number = struct.unpack('>i', magic_number)[0]

number_of_images = f.read( 4 )
number_of_images = struct.unpack('>i', number_of_images)[0]

number_of_rows = f.read( 4 )
number_of_rows = struct.unpack('>i', number_of_rows)[0]

number_of_columns = f.read( 4 )
number_of_columns = struct.unpack('>i', number_of_columns)[0]

bytes_per_image = number_of_rows * number_of_columns

l_magic_number = l.read(4)
l_magic_number = struct.unpack('>i', l_magic_number)[0]
l_number_of_images = l.read(4)
l_number_of_iamges = struct.unpack('>i', l_number_of_images)[0]


for num in range(60000):
    raw_img = f.read(bytes_per_image)
    format = '%dB' % bytes_per_image
    lin_img = struct.unpack(format, raw_img)
    np_ary = np.asarray(lin_img).astype('uint8')
    np_ary = np.reshape(np_ary, (28,28),order='C')

    label_byte = l.read(1)
    label_int = int.from_bytes(label_byte, byteorder='big')

    path = os.path.join(PATH, str(label_int))
    if not os.path.isdir(path):
        print(f"make dir {label_int}")
        os.makedirs(path)

    pil_img = Image.fromarray(np_ary)
    pil_img.save(os.path.join(path, str(label_int)+ "_" + str(num) + ".png"))
    print(f"save image {label_int} in {path}")

