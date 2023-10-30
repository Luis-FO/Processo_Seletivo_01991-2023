import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt


def noise_remove(im_path:str, im_gray:str):
    # Carrega as imagens
    im = cv.imread(im_path, cv.IMREAD_GRAYSCALE)
    im_gray = cv.imread(im_gray, cv.IMREAD_GRAYSCALE)

    # Aplica medianBlur com tamanho 5 para eliminar o ruído
    MedianB = cv.medianBlur(im, 5)

    # Configura exibição das imagens
    fig=plt.figure(dpi=300)
    fig.add_subplot(1,3,1)
    plt.imshow(im_gray,cmap='gray')
    plt.axis("off")
    plt.title("Original")

    fig.add_subplot(1,3,2)
    plt.imshow(im,cmap='gray')
    plt.axis("off")
    plt.title("com ruído")

    fig.add_subplot(1,3,3)
    plt.imshow(MedianB,cmap='gray')
    plt.axis("off")
    plt.title("MedianB")

    # Salva imagem tratada
    cv.imwrite("Q1/result/solution_q1.png", MedianB)

    # Exibe imagens na tela
    plt.show()

if __name__ == "__main__":
    im_path = "Q1/Imagens/img46_gray_noise.png"
    im2 = "Q1/Imagens/img46_gray.png"
    noise_remove(im_path, im2)