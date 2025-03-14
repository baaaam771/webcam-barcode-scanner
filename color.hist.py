import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

def select_image():
    """사용자가 파일을 선택하도록 하는 함수"""
    root = Tk()
    root.withdraw()  # Tk 창 숨기기
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg;*.bmp;*.tiff")])
    return file_path

def plot_histogram(image_path):
    """이미지의 그레이스케일 히스토그램을 출력하는 함수"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 이미지를 그레이스케일로 불러오기
    
    if image is None:
        print("이미지를 불러올 수 없습니다.")
        return
    
    # 히스토그램 계산
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # 히스토그램 그래프 출력
    plt.figure(figsize=(10, 5))
    plt.plot(hist, color='black')
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.grid()
    plt.show()

if __name__ == "__main__":
    image_path = select_image()
    if image_path:
        plot_histogram(image_path)