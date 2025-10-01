import os
from skimage.segmentation import slic
from skimage import io, color, filters
import numpy as np
import cv2
from skimage.measure import find_contours
import matplotlib.pyplot as plt

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

def close_holes(binary):
    ''' Fecha buracos na máscara binária usando morfologia matemática '''
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary.astype('uint8'), cv2.MORPH_CLOSE, kernel, iterations=3)
    return closed

def background_removal(img, binary):
    ''' Remove o fundo da imagem usando a máscara binária '''
    iterations = 20
    margem = 5
    binary1 = cv2.dilate(binary.astype('uint8'), None, iterations=iterations)
    binary2 = cv2.erode(binary1, None, iterations=iterations - margem)
    mask = binary2.astype('uint8') * 255
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

BASE_DIR = "./Trabalho_2"
if not os.path.exists(BASE_DIR):
    BASE_DIR = "."
print(f"Using base dir: {BASE_DIR}")

BASE_PATH = f"{BASE_DIR}/fotos/"
if not os.path.exists(BASE_PATH):
    BASE_PATH = f"{BASE_DIR}/fotos/"
print(f"Using base path: {BASE_PATH}")

nomes_cenas = sorted(os.listdir(BASE_PATH))
print(nomes_cenas)

for nome_cena in nomes_cenas:
    CENARIO = nome_cena

    IMAGES_PATH = BASE_PATH + CENARIO + "/"
    OUTPUT_SEG_PATH =  f"{BASE_DIR}/seg_fotos/{CENARIO}/"
    OUTPUT_MASK_PATH =  f"{BASE_DIR}/mask_fotos/{CENARIO}/"

    os.makedirs(OUTPUT_SEG_PATH, exist_ok=True)
    os.makedirs(OUTPUT_MASK_PATH, exist_ok=True)

    # Processa todas as imagens na pasta
    image_names = os.listdir(IMAGES_PATH)
    print(f"Processing scenario: {CENARIO} with {len(image_names)} images.")
    k = 0
    for name in image_names:
        if name.startswith('.'):
            continue
        img = io.imread(IMAGES_PATH + name)
        binary = segmentate(img)
        result = background_removal(img, binary)
        io.imsave(OUTPUT_MASK_PATH + name, close_holes(binary).astype('uint8') * 255)
        io.imsave(OUTPUT_SEG_PATH + name, result)
        k += 1
        if k % 10 == 0:
            print(f"Processed {k}/{len(image_names)} images...")
    print(f"Finished processing scenario: {CENARIO}. Total images processed: {k}.")
print("All scenarios processed.")