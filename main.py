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
BACKGROUND = 'img/bg.bmp'

def smooth_threshold(value, min, max):
    if value < min:
        return 0.0
    elif value > max:
        return 1.0
    else:
        return (value - min)/(max - min)

#def color_mask_rgb(r, g, b, r0, g0, b0, min_threshold, max_threshold):
#    distance = np.sqrt((r - r0)**2 + (g - g0)**2 + (b - b0)**2)
#    return smooth_threshold(distance, min_threshold, max_threshold)

#def calculate_mask_rgb(img, r0, g0, b0, min_threshold, max_threshold):
#    mask = img[:, :, 0].astype(np.float32)
#    h, w = mask.shape
#    for y in range(h):
#        for x in range(w):
#            mask[y, x] = color_mask_rgb(img[y, x, 2], img[y, x, 1], img[y, x, 0], r0, g0, b0, min_threshold, max_threshold)
#    return mask

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

def supress_foreground_key(img, hue_key, min_key_diff, max_key_diff):
    opposite_key = (hue_key + 180) % 360
    img = img.astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, c = img.shape

    cos = np.zeros_like(hsv[:, :, 2]).astype(np.float64)

    for y in range(h):
        for x in range(w):
            angle = (np.abs(hsv[y, x, 0] - hue_key) + 180) % 360 - 180
            alpha = 1.0 - smooth_threshold(angle, min_key_diff, max_key_diff)
            angle_cos = np.cos(np.deg2rad(angle)) * alpha
            cos[y, x] = angle_cos
            if angle_cos < 0.0:
                angle_cos = 0.0
                cos[y, x] = 0.0
            hsv[y, x, 0] = opposite_key
            hsv[y, x, 2] = hsv[y, x, 2] * angle_cos
    
    supressed = 0.5 * img + 0.5 * cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    supressed[:, :, :] = supressed[:, :, :] * (2.0 - cos)[:, :, None]

    return supressed.astype(np.float32)

def main():
    background = cv2.imread(BACKGROUND, cv2.IMREAD_COLOR)
    for nome, img in INPUT_IMAGES:
        img = cv2.imread(img, cv2.IMREAD_COLOR).astype(np.float32)
        if img is None:
            print('Failed to open image. \n')
            sys.exit()

        bg = cv2.resize(background, (img.shape[1], img.shape[0]), cv2.INTER_LINEAR)

        #print("Green:", cv2.cvtColor(np.uint8([[[0, 255, 0]]]), cv2.COLOR_BGR2YCrCb)/255.0)

        mask = calculate_mask_ycbcr(img, 0.25, 0.25, 0.2, 0.25)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
        inverted_mask = 1.0 - mask

        supressed = supress_foreground_key(img, 120.0, 55.0, 65.0)

        border_mask = mask
        #border_mask = np.where(mask < 1.0, mask, 0.0)
        #border_mask = cv2.GaussianBlur(border_mask, (0, 0), 3.0)

        img = border_mask[:, :, None] * supressed + (1.0 - border_mask)[:, :, None] * img

        hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)

        value_mask = hsv[:, :, 2]/255.0 * inverted_mask

        img[:, :, :] = img[:, :, :] * mask[:, :, None] + bg[:, :, :] * value_mask[:, :, None]
        
        cv2.imwrite(f'out/{nome}-mask.bmp', mask*255)
        #cv2.imwrite(f'out/{nome}-bmask.bmp', border_mask*255)
        cv2.imwrite(f'out/{nome}-vmask.bmp', value_mask*255)
        cv2.imwrite(f'out/{nome}.bmp', img)

        print(f'out/{nome}')

if __name__ == '__main__':
    main()