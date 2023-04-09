import cv2
import imutils
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'Data/OCR/tesseract.exe'

def Pretreatment(images):
    img = cv2.imread(images)
    img = imutils.resize(img, width=400)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bilateralFilter(img_gray, 13, 15, 15)
    edged = cv2.Canny(img_gray, 170, 200)
    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image = img.copy()
    contours = imutils.grab_contours(contours)
    cv2.drawContours(image, contours, -1, (0, 255, 255), 3)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCount = None
    Top_Image = image.copy()
    cv2.drawContours(Top_Image, contours, -1, (0, 255, 0), 3)
    for i in contours:
        perimeter = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)
        if (len(approx) == 4):
            NumberPlateCount = approx
            x, y, w, h = cv2.boundingRect(i)
            crp_img = img[y:y + h, x:x + w]
            crp_img = imutils.resize(crp_img, width=200)
            cv2.imwrite('IMAGE.png', crp_img)
            cv2.drawContours(img, [NumberPlateCount], -1, (0, 255, 0), 3)
            cv2.imwrite('Find_the_License_Plate.png', img)
            break

def red_Image():
    img = cv2.imread('IMAGE.png')
    text = pytesseract.image_to_string(img, lang="eng", config='--psm 6')
    # print('Kết quả nhận về là: ', text)
    filtered_text = re.findall("[a-zA-Z0-9]+", text)  # Lọc ra chỉ những chữ hoặc số
    texts = "".join(filtered_text)
    print('Kết quả tìm ra là: ', texts)  # In kết quả đã lọc ra
    return texts

def red_CAMERA():
    img = cv2.imread('ROI.png')
    text = pytesseract.image_to_string(img, lang="eng", config='--psm 6')
    # print('Kết quả nhận về là: ', text)
    filtered_text = re.findall("[a-zA-Z0-9]+", text)  # Lọc ra chỉ những chữ hoặc số
    texts = "".join(filtered_text)
    print('Kết quả video tìm ra là: ', texts)  # In kết quả đã lọc ra
    return texts


red_CAMERA()