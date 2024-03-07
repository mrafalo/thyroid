import numpy as np       
import cv2
import skimage.segmentation
from skimage import img_as_ubyte, exposure

def view_image(_im):
    cv2.imshow("view", _im)
    cv2.waitKey(0)

def view_images(_im1, _im2):
    cv2.imshow("before", _im1)
    cv2.imshow("after", _im2)
    cv2.waitKey(0)

def felzenszwalb(_image, _k):
    
    res = skimage.segmentation.felzenszwalb(_image, scale = _k)
    return cv2.cvtColor(img_as_ubyte(exposure.rescale_intensity(res)), cv2.COLOR_GRAY2BGR)
    
def drawMask(_image, _cnts, fill=True):
    image = np.array(_image)
    markers = np.zeros((image.shape[0], image.shape[1]))
    heatmap_img = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    t = 2
    if fill:
      t = -1
    cv2.drawContours(markers, _cnts, -1, (255, 0, 0), t)
    mask = markers>0
    image[mask,:] = heatmap_img[mask,:]
    return image


def heatmap(_image):

    return cv2.applyColorMap(_image, cv2.COLORMAP_RAINBOW)
    
def edges(_image, _lower, _upper):
    img = cv2.Canny(_image, _lower, _upper, L2gradient = True )
    return img
    
def sobel(_im, _dx=1, _dy=1, _ksize=3):
    
    gray = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=3, scale=1)
    y = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=3, scale=1)
    absx= cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    res = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
    return res# cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)


def view_images_matrix(_im_list):
    res = cv2.vconcat([cv2.hconcat(list_h) for list_h in _im_list])
    # image resizing
    res = cv2.resize(res, dsize = (0,0), fx = 0.3, fy = 0.3)

    # show the output image
    cv2.imshow('images', res)
    cv2.waitKey(0)
        

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def get_images_matrix(_im_list):

    return vconcat_resize_min(_im_list)
        
    
def dilate(_im):
    
    if len(_im.shape) == 3:
        gray = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
    else:
        gray = _im
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (70,1))
    res = cv2.dilate(gray, horizontal_kernel, iterations=1)

    return cv2.cvtColor(res, cv2.COLOR_GRAY2BGR) 

def bw_mask(_im):
    # convert to gray
    if len(_im.shape) == 3:
        gray = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
    else:
        gray = _im
    # mask = np.zeros(_im.shape, dtype=np.uint8)  
    # mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    # res = cv2.bitwise_and(_im, _im, mask = mask)
    # #res[mask==0] = (255,255,255)
    (thresh, res) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #thresh = 127
    #res = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    return res 

def threshold(_im):
    
    # convert to gray
    if len(_im.shape) == 3:
        gray = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
    else:
        gray = _im
    
    # create mask
    #thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    
    # dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    res = _im.copy()
    res[mask == 255] = (255,255,255)
    
    return res 

def blur(_im):
    
    if len(_im.shape) == 3:
        gray = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
    else:
        gray = _im
        
    # create mask
    thresh = cv2.threshold(gray, 247, 255, cv2.THRESH_BINARY)[1]
    
    # dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    blur = cv2.blur(_im,(5,5),0)
    res = _im.copy()
    res[mask>0] = blur[mask>0]

    #res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) 
    return res
    
def blur_manual(_im, _blur_range=1, _border=5):
    
    
    if len(_im.shape) == 3:
        gray = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
    else:
        gray = _im

    
    # create mask
    thresh = cv2.threshold(gray, 247, 255, cv2.THRESH_BINARY)[1]
    
    # dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    
    res = _im.copy()
    res[mask == 255] = (255,255,255)
    
    w_min = _border
    w_max = _im.shape[1] - _border
    
    h_min = _border
    h_max = _im.shape[0] - _border 
    
    if len(_im.shape) == 3:
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) 
    
    for row in range(h_min, h_max):
        for col in range(w_min, w_max):
            if res[row][col] == 255:
                tmp_arr = res[row -_blur_range:row + _blur_range, col - _blur_range:col + _blur_range][res[row - _blur_range:row + _blur_range, col - _blur_range:col + _blur_range] < 255]
              
                if tmp_arr.size>0:
                    tmp = np.mean(tmp_arr)
                else:
                    tmp = np.mean(res)
                    
                res[row][col] = round(np.mean(tmp))
      
    return res 

