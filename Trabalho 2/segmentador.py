import os
from skimage.segmentation import slic
from skimage import io, color, filters
import numpy as np
import cv2
from skimage.measure import find_contours
import matplotlib.pyplot as plt

BASE_PATH = "./Trabalho 2/fotos/"
CENARIO = "cenario4-vaso/"
IMAGES_PATH = BASE_PATH + CENARIO
OUTPUT_PATH =  f"./Trabalho 2/seg_fotos/{CENARIO}"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def segmentate(img):
    ''' Segmenta a imagem em superpixels e retorna a imagem segmentada e uma máscara binária '''
    
    segments = slic(img, n_segments=200, compactness=15, sigma=1, start_label=1)
    print(f"Number of segments: {len(set(segments.flatten()))}")
    summary = color.label2rgb(segments, img, kind='avg')
    thresh = filters.threshold_otsu(color.rgb2gray(summary))
    binary = color.rgb2gray(summary) < thresh

    return summary, binary, segments

def green_superpixel_mask(img, segments):
    '''Cria uma máscara booleana onde True indica superpixels verdes'''
    mask = np.zeros(segments.shape, dtype=bool)
    for label in np.unique(segments):
        sp_mask = segments == label
        mean_color = img[sp_mask].mean(axis=0)
        # Critério simples para verde: canal G maior que R e B, e G acima de um limiar
        if mean_color[1] > 20 and \
            mean_color[1] > mean_color[0] + 20 and \
            mean_color[1] > mean_color[2] + 20:
            mask[sp_mask] = True
    return mask

def background_removal(img, binary):
    ''' Remove o fundo da imagem usando a máscara binária '''
    binary1 = cv2.dilate(binary.astype('uint8'), None, iterations=100)
    binary2 = cv2.erode(binary1, None, iterations=95)
    mask = binary2.astype('uint8') * 255
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

# Processa todas as imagens na pasta
image_names = os.listdir(IMAGES_PATH)
for name in image_names:
    if name.startswith('.'):
        continue
    img = io.imread(IMAGES_PATH + name)
    summary, binary, segments = segmentate(img)
    # green_mask = green_superpixel_mask(img, segments)
    result = background_removal(img, binary)
    # result = background_removal(img, green_mask)
    io.imsave(OUTPUT_PATH + name, result)

# # Processa uma imagem específica e exibe os resultados
# IMAGE_PATH = "./Trabalho 2/fotos/cenario4-vaso/IMG_0456.jpeg"

# img = io.imread(IMAGE_PATH)
# summary, binary, segments = segmentate(img)
# # green_mask = green_superpixel_mask(img, segments)
# result = background_removal(img, binary)
# # result = background_removal(img, green_mask)

# plt.figure(figsize=(12, 16))
# plt.subplot(2, 2, 1)
# plt.title('Original Image')
# plt.imshow(img)
# plt.axis('off')

# plt.subplot(2, 2, 2)
# plt.title('Segmented Image')
# plt.imshow(summary)
# # plt.contour(segments, colors='red', linewidths=1)
# plt.axis('off')

# plt.subplot(2, 2, 3)
# plt.title('Binary Mask')
# plt.imshow(binary, cmap='gray')
# plt.axis('off')

# # plt.subplot(2, 2, 3)
# # plt.title('Green Mask')
# # plt.imshow(green_mask, cmap='gray')
# # plt.axis('off')

# plt.subplot(2, 2, 4)
# plt.title('Background Removed')
# plt.imshow(result, cmap='gray')
# plt.axis('off')

# plt.show()