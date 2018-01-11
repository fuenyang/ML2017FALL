import numpy as np
import skimage
from skimage import io
import sys
import os


def imgFlatten(image):
    image_f = []
    for i in range(len(image)):
        tmp = image[i]
        img_f = tmp.flatten()
        image_f.append(img_f)

    image_f = np.array(image_f)
    return image_f

def testImage(filename):
    img = io.imread(filename)
    img = img.flatten()

    return img

def imgReshape(image):
    image -= np.min(image)
    image /= np.max(image)
    image = (image*255).astype(np.uint8)

    image = np.reshape(image,(600,600,3))

    return image



dirs = os.listdir(sys.argv[1])



images = []
for filename in dirs:
    filename = os.path.join(sys.argv[1],filename)
    ima = io.imread(filename)
    images.append(ima)
images = np.array(images)





images_f = imgFlatten(images)


aveface = images_f.mean(axis=0)


images_fmm = images_f - aveface


u, s, v = np.linalg.svd(images_fmm.T, full_matrices=False)

# print('u:',u.shape)
# print('v',v.shape)


# reconstruction
testfilename = os.path.join(sys.argv[1],sys.argv[2])



image_test = testImage(testfilename)

image_test = image_test-aveface

weight = np.dot(image_test,u[:,:4])


reimage = np.zeros(1080000)
for i in range(len(weight)):
    reimage = reimage+weight[i]*u[:,i]

reimage = reimage+aveface


reimage = imgReshape(reimage)
aveface = imgReshape(aveface)
# eig1 = imgReshape(-u[:,0])
# eig2 = imgReshape(-u[:,1])
# eig3 = imgReshape(-u[:,2])
# eig4 = imgReshape(-u[:,3])
# eig10 = imgReshape(-u[:,9])

# print(reimage.shape)
io.imsave('reconstruction.jpg',reimage)


# print('eig1=',s[0]/s.sum())
# print('eig2=',s[1]/s.sum())
# print('eig3=',s[2]/s.sum())
# print('eig4=',s[3]/s.sum())










