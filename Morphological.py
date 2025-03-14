# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def preprocess_image(image_path):
#     # 이미지 로드 (그레이스케일 변환)
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # 가우시안 블러 적용 (노이즈 감소)
#     blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
#     # OTSU 이진화
#     _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
#     # 커널 생성 (모폴로지 연산에 사용)
#     kernel = np.ones((3, 3), np.uint8)
    
#     # 모폴로지 연산 적용 (열림 연산으로 작은 노이즈 제거)
#     morph_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
#     # 팽창 적용 (바코드 막대 더 강조)
#     dilated = cv2.dilate(morph_open, kernel, iterations=2)
    
#     return image, binary, morph_open, dilated

# def display_results(image_path):
#     # 이미지 변환 실행
#     original, binary, morph_open, dilated = preprocess_image(image_path)
    
#     # 결과 출력
#     titles = ['Original', 'Binary', 'Morph Open', 'Dilated']
#     images = [original, binary, morph_open, dilated]
    
#     plt.figure(figsize=(10, 5))
#     for i in range(4):
#         plt.subplot(1, 4, i+1)
#         plt.imshow(images[i], cmap='gray')
#         plt.title(titles[i])
#         plt.axis('off')
#     plt.show()

# # 이미지 파일 경로 입력 (사용자가 지정해야 함)
# image_path = 'cropped_region_0.png'  # 바코드가 있는 이미지 경로

# # 실행
# display_results(image_path)



# import cv2 as cv
# import sys

# img=cv.imread('cropped_region_0.png')

# t, bin_img = cv. threshold(img[:,:,2], 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
# print('오츄 알고리즘이 찾은 임계값=', t)

# cv.imshow('R channel', img[:, :, 2])
# cv.imshow('R channel', bin_img)

# cv.waitKey()
# cv.destroyAllWindows()


import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread('cropped_region_0.png', cv2.IMREAD_GRAYSCALE)
# src = cv2.imread('blur_barcode.png', cv2.IMREAD_GRAYSCALE)



if src is None:
    print('Image load failed!')
    sys.exit()

# src 영상에 지역 이진화 수행
dst1 = np.zeros(src.shape, np.uint8)

bw = src.shape[1] // 4
bh = src.shape[0] // 4

for y in range(4):
    for x in range(4):
        src_ = src[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
        dst_ = dst1[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
        cv2.threshold(src_, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU, dst_)
        
# 영상 안에 있는 흰색 덩어리를 정수 형태로 리턴
cnt1, _ = cv2.connectedComponents(dst1)
print('cnt1:', cnt1)

# 모폴로지 열기
dst2 = cv2.morphologyEx(dst1, cv2.MORPH_OPEN, None)

# 영상 안에 있는 흰색 덩어리를 정수 형태로 리턴
cnt2, _ = cv2.connectedComponents(dst2)
print('cnt2:', cnt2)

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()