import os
from skimage.segmentation import slic
from skimage import io, color, filters
import numpy as np
import cv2
from skimage.measure import find_contours
import matplotlib.pyplot as plt

BASE_PATH = "./Trabalho 2/fotos/"
CENARIO = "boneco2"  # Altere para o cenário desejado
IMAGES_PATH = BASE_PATH + CENARIO + "/"
OUTPUT_PATH =  f"./Trabalho 2/seg_fotos/{CENARIO}/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# def green_superpixel_mask(img, segments):
#     '''Cria uma máscara booleana onde True indica superpixels verdes'''
#     mask = np.zeros(segments.shape, dtype=bool)
#     for label in np.unique(segments):
#         sp_mask = segments == label
#         mean_color = img[sp_mask].mean(axis=0)
#         # Critério simples para verde: canal G maior que R e B, e G acima de um limiar
#         if mean_color[1] > 20 and \
#             mean_color[1] > mean_color[0] + 20 and \
#             mean_color[1] > mean_color[2] + 20:
#             mask[sp_mask] = True
#     return mask

def segmentate_old(img):
    ''' Segmenta a imagem em superpixels e retorna a imagem segmentada e uma máscara binária '''
    
    segments = slic(img, n_segments=200, compactness=15, sigma=1, start_label=1)
    print(f"Number of segments: {len(set(segments.flatten()))}")
    summary = color.label2rgb(segments, img, kind='avg')
    thresh = filters.threshold_otsu(color.rgb2gray(summary))
    binary = color.rgb2gray(summary) < thresh

    return summary, binary, segments

def segmentate(img):
    ''' Converte a imagem para o espaço LAB e segmenta via thresholding do canal A '''
    img_lab = color.rgb2lab(img)
    a_channel = img_lab[:, :, 1]
    thresh_a = filters.threshold_otsu(a_channel)
    g_mask = a_channel > thresh_a
    l_mask = a_channel < thresh_a

    if g_mask.sum() > l_mask.sum():
        binary = l_mask
    else:
        binary = g_mask
    return binary

def background_removal(img, binary):
    ''' Remove o fundo da imagem usando a máscara binária '''
    iterations = 20
    margem = 5
    binary1 = cv2.dilate(binary.astype('uint8'), None, iterations=iterations)
    binary2 = cv2.erode(binary1, None, iterations=iterations - margem)
    mask = binary2.astype('uint8') * 255
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

# Processa todas as imagens na pasta
image_names = os.listdir(IMAGES_PATH)
for name in image_names:
    if name.startswith('.'):
        continue
    img = io.imread(IMAGES_PATH + name)
    binary = segmentate(img)
    result = background_removal(img, binary)
    io.imsave(OUTPUT_PATH + name, result)

# # Processa uma imagem específica e exibe os resultados
# # IMAGE_PATH = "./Trabalho 2/fotos/cenario4-vaso/IMG_0505.jpeg"
# # IMAGE_PATH = "./Trabalho 2/fotos/cenario5-bulbassauro/IMG_0668.jpeg"
# # IMAGE_PATH = "./Trabalho 2/fotos/cenario3-dinofauro/IMG_0433.jpeg"
# IMAGE_PATH = "./Trabalho 2/fotos/boneco-compressed/IMG_20250924_110307444.jpg"

# img = io.imread(IMAGE_PATH)

# img_lab = color.rgb2lab(img)
# binary = segmentate(img)


# a_channel = img_lab[:, :, 1]
# thresh_a = filters.threshold_otsu(a_channel)
# g_mask = a_channel > thresh_a
# l_mask = a_channel < thresh_a

# if g_mask.sum() > l_mask.sum():
#     binary = l_mask
# else:
#     binary = g_mask

# result = background_removal(img, binary)


# plt.figure(figsize=(18, 16))

# # Exibe os canais L, A, B, a imagem original, a máscara binária e a imagem com fundo removido
# plt.subplot(2, 3, 1)
# plt.title('L Channel')
# plt.imshow(img_lab[:, :, 0], cmap='gray')
# plt.axis('off')

# plt.subplot(2, 3, 2)
# plt.title('A Channel')
# plt.imshow(img_lab[:, :, 1], cmap='gray')
# plt.axis('off')

# plt.subplot(2, 3, 3)
# plt.title('B Channel')
# plt.imshow(img_lab[:, :, 2], cmap='gray')
# plt.axis('off')

# plt.subplot(2, 3, 4)
# plt.title('Original Image')
# plt.imshow(img)
# plt.axis('off')

# plt.subplot(2, 3, 5)
# plt.title('Binary Mask')
# plt.imshow(binary, cmap='gray')
# plt.axis('off')

# plt.subplot(2, 3, 6)
# plt.title('Background Removed')
# plt.imshow(result, cmap='gray')
# plt.axis('off')

# plt.show()