from imgaug import augmenters as iaa
import imgaug as ia
import cv2




def get_augmenter(name):

    alot = lambda aug: iaa.Sometimes(0.80, aug)
    sometimes = lambda aug: iaa.Sometimes(0.50, aug)
    few = lambda aug: iaa.Sometimes(0.20, aug)

    if 'rgb' in name:
        seq_rgb = iaa.Sequential([

            iaa.Fliplr(0.25),  # horizontally flip 50% of the images
            iaa.Flipud(0.25),  # horizontally flip 50% of the images
            sometimes(iaa.Add((-30, 30))),
            sometimes(iaa.Multiply((0.80, 1.20), per_channel=False)),
            sometimes(iaa.GaussianBlur(sigma=(0, 0.20))),
            sometimes(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.00, 0.20), per_channel=0.5)),
            sometimes(iaa.ContrastNormalization((0.7, 1.4))),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.20, 0.2), "y": (-0.2, 0.2)},
                # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                order=1,  #bilinear interpolation (fast)
                cval=0,
                mode="constant"
                # cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                # mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            ))])
        return seq_rgb

    else:

        seq_multi = iaa.Sequential([

            sometimes(iaa.Affine(
                # scale images to 80-120% of their size, individually per axis
                # translate by -20 to +20 percent (per axis)
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.20, 0.2), "y": (-0.2, 0.2)},
                # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                order=0,  # use nearest neighbour
                cval=127.5,
                mode="constant"
            ))
        ])
        return seq_multi
