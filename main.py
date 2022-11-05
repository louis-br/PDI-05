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

def transfer_hue_from_bg(img, bg, mask, min_threshold, max_threshold):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    print(f'Transfer hue, min: {np.min(img[:, :, 0])}, max: {np.max(img[:, :, 0])}')
    h, w, c = img.shape
    for y in range(h):
        for x in range(w):
            strength = mask[y, x]
            if mask[y, x] > min_threshold and mask[y, x] < max_threshold:
                img[y, x, 0] = img[y, x, 0] * (1.0 - strength) + bg[y, x, 0] * strength
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

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
    background = cv2.imread(BACKGROUND, cv2.IMREAD_COLOR)
    for nome, img in INPUT_IMAGES:
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        if img is None:
            print('Failed to open image. \n')
            sys.exit()

        bg = cv2.resize(background, (img.shape[1], img.shape[0]), cv2.INTER_LINEAR)

        #mask = calculate_mask_rgb(img, 84.0, 166.0, 66.0, 16.0, 128.0)

        mask = calculate_mask_ycbcr(img, 0.25, 0.25, 0.2, 0.3)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
        #mask = np.where(mask == 1.0, 0.0, mask)
        inverted_mask = 1.0 - mask


        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        value_mask = hsv[:, :, 2]/255.0 * inverted_mask

        img = transfer_hue_from_bg(img, bg, value_mask, 0.0, 1.1)
        #mean = np.ma.masked_where(inverted_mask > 0.9, value_mask).max() #.mean()#np.mean(value_mask)
        #print(mean)

        #value_mask = value_mask - mean

        print(f'vmask min: {np.min(value_mask)} max: {np.max(value_mask)}')

        #for c in range(3):
        #    img[:, :, c] = img[:, :, c] * mask + hsv[:, :, 2] * inverted_mask

        #cv2.imwrite(f'out/{nome}-sat.bmp', hsv[:, :, 1])
        #cv2.imwrite(f'out/{nome}-val.bmp', hsv[:, :, 2])


        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        #cv2.imwrite(f'out/{nome}-lum.bmp', hls[:, :, 1])
        #cv2.imwrite(f'out/{nome}-sat-hls.bmp', hls[:, :, 2])



        #mask_hsv = calculate_mask_hsv(img, 60, 15, 25, 128.0, 128.0)

        for c in range(3):
            img[:, :, c] = img[:, :, c] * mask + bg[:, :, c] * value_mask #(value_mask + bg[:, :, c]) * inverted_mask # + img[:, :, c] * inverted_mask * 0.25
        
        cv2.imwrite(f'out/{nome}-mask.bmp', mask*255)
        cv2.imwrite(f'out/{nome}-vmask.bmp', value_mask*255)
        #cv2.imwrite(f'out/{nome}-mask-hsv.bmp', mask_hsv*255)
        cv2.imwrite(f'out/{nome}.bmp', img)

if __name__ == '__main__':
    main()