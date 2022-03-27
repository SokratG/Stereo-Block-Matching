import matplotlib.pyplot as plt
import cv2
from StereoBlockMatcher import StereoBlockMatcher


def show_imgs(im_left, im_right, cmap='jet'):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))

    ax1.imshow(im_left, cmap=cmap, interpolation='bilinear')
    ax1.title.set_text('Image1')
    ax1.autoscale(False)
    ax1.set_axis_on()

    ax2.imshow(im_right, cmap=cmap, interpolation='bilinear')
    ax2.title.set_text('Image2')
    ax2.autoscale(False)
    ax2.set_axis_on()

    plt.show()
    return


def show_img(img, cmap='jet', name='noname'):
    plt.figure()
    plt.imshow(img, cmap=cmap,  interpolation='bilinear')
    plt.title(name)
    plt.colorbar()
    plt.show()
    return


def test():
    base_path = './data/adirondack/'
    im_left = cv2.imread(base_path + 'im_left.png', cv2.IMREAD_UNCHANGED)
    im_right = cv2.imread(base_path + 'im_right.png', cv2.IMREAD_UNCHANGED)

    im_left = cv2.resize(im_left, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
    im_right = cv2.resize(im_right, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
    
    stereo = StereoBlockMatcher(block_size=5, max_search_range=25)
    #res = stereo.stereo_compute(im_left, im_right, use_subpixel_refine=True)
    res = stereo.stereo_dynamic_compute(im_left, im_right, 0.55)

    show_img(res, cmap='jet', name='Disparity map')
    return
    

if __name__ == '__main__':
    test()
