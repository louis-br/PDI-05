import sys
import numpy as np
import cv2

INPUT_IMAGES = (
    ('0', 'img/0.BMP'),
    ('1', 'img/1.bmp'),
    ('2', 'img/2.bmp'),
    ('3', 'img/3.bmp'),
    ('4', 'img/4.bmp'),
    ('5', 'img/5.bmp'),
    ('6', 'img/6.bmp'),
    ('7', 'img/7.bmp'),
    ('8', 'img/8.bmp'),
)

def smooth_threshold(value, min, max):
    if value < min:
        return 0.0
    elif value > max:
        return 1.0
    else:
        return (value - min)/(max - min)

def color_mask_rgb(r, g, b, r0, g0, b0, min_threshold, max_threshold):
    distance = np.sqrt((r - r0)**2 + (g - g0)**2 + (b - b0)**2)
    return smooth_threshold(distance, min_threshold, max_threshold)

def calculate_mask_rgb(img, r0, g0, b0, min_threshold, max_threshold):
    mask = img[:, :, 0].astype(np.float32)
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            mask[y, x] = color_mask_rgb(img[y, x, 2], img[y, x, 1], img[y, x, 0], r0, g0, b0, min_threshold, max_threshold)
    return mask

def color_mask_ycbcr(y, cb, cr, cb0, cr0, min_threshold, max_threshold):
    distance = np.sqrt((cb - cb0)**2 + (cr - cr0)**2)
    return smooth_threshold(distance, min_threshold, max_threshold)

def calculate_mask_ycbcr(img, cb0, cr0, min_threshold, max_threshold):
    mask = img[:, :, 0].astype(np.float32)
    img = cv2.cvtColor(img.astype(np.float32)/255.0, cv2.COLOR_BGR2YCrCb)
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            mask[y, x] = color_mask_ycbcr(img[y, x, 0], img[y, x, 2], img[y, x, 1], cb0, cr0, min_threshold, max_threshold)
    return mask

#def color_mask_hue(hue, saturation, value, key, min_range, max_range, min_saturation, min_value):
#    diff = np.abs(hue - key)
#    return smooth_threshold(diff, min_range, max_range)# * \
#           #smooth_threshold(saturation, min_saturation, min_saturation + 10) * \
#           #smooth_threshold(value, min_value, min_value + 10)

#def calculate_mask_hsv(img, key, min_range, max_range, min_saturation, min_value):
#    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#    mask = img[:, :, 0].astype(np.float32)
#    h, w = mask.shape
#    for y in range(h):
#        for x in range(w):
#            mask[y, x] = color_mask_hue(hsv[y, x, 0], hsv[y, x, 1], hsv[y, x, 2], key, min_range, max_range, min_saturation, min_value)
#    return mask

def main():
    for nome, img in INPUT_IMAGES:
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        if img is None:
            print('Failed to open image. \n')
            sys.exit()

        #mask = calculate_mask_rgb(img, 0.0, 226.0, 13.0, 16.0, 128.0)

        mask = calculate_mask_ycbcr(img, 0.25, 0.25, 0.2, 0.25)

        #mask_hsv = calculate_mask_hsv(img, 60, 15, 25, 128.0, 128.0)

        for c in range(3):
            img[:, :, c] = img[:, :, c] * mask
        
        cv2.imwrite(f'out/{nome}-mask.bmp', mask*255)
        #cv2.imwrite(f'out/{nome}-mask-hsv.bmp', mask_hsv*255)
        cv2.imwrite(f'out/{nome}.bmp', img)

if __name__ == '__main__':
    main()