import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob
from sklearn.cluster import KMeans


def region_selector(im_dir:str):
    # Lê diretório
    path_list = glob.glob(im_dir+'/*.png')

    fig=plt.figure(dpi=300)
    for i in range(len(path_list)):
        # Escolhe imagem
        im_path = path_list[i]

        # Carrega imagem em um array
        im = cv.imread(im_path)
        im = cv.cvtColor(im, cv.COLOR_BGR2HSV)

        # Aplica o Blur
        MedianB = cv.medianBlur(im, 15)

        # Normaliza e redimensiona para
        im_resized = MedianB.reshape((MedianB.shape[1]*MedianB.shape[0],3))

        kmeans= KMeans(n_clusters=15, max_iter=1000, algorithm='auto', random_state=0)
        s = kmeans.fit(im_resized)

        labels = kmeans.labels_
        print(min(labels))
        labels = list(labels)

        centroid = kmeans.cluster_centers_
        print(centroid)

        result = np.uint8(kmeans.cluster_centers_[kmeans.labels_])
        result2 = result.reshape((MedianB.shape))
        
        green_lower = np.array([15, 66, 59], np.uint8)
        green_upper = np.array([19, 92, 73], np.uint8)

        mask2 = cv.inRange(result2, green_lower, green_upper)
        kernel = np.ones((15, 15), np.uint8) 
        image = cv.erode(mask2, kernel, cv.BORDER_REFLECT)
        image = cv.dilate(image, kernel, cv.BORDER_REFLECT)
        fig.add_subplot(2,2,i+1)
        plt.axis("off")
        plt.title(f"Imagem {i}")
        plt.imshow(image)
    
    plt.show()
    

if __name__ == "__main__":
    im_path = "Q2/Imagens"
    region_selector(im_path)