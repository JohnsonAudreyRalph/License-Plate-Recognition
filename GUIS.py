import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.models import load_model
from keras.utils import custom_object_scope
from keras.metrics import Precision, Recall

root = tk.Tk()
root.title("Nhận diện biển số xe tự động")
root.geometry("850x550")
root.configure()
Show_Image = tk.Label(root)
Label_Predicted_Results = tk.Label(root, font=('arial', 15, 'bold'))

def custom_f1score(y_true, y_pred):
    precision = Precision()(y_true, y_pred)
    recall = Recall()(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + 1e-8))

with custom_object_scope({'custom_f1score': custom_f1score}):
    model = load_model('Data/Save_mode.hdf5')

def points(location):
    comparelist = []
    sortedlist = []
    for point in location:
        comparelist.append(np.sum(point))
    for i in range(4):
        max_pos = comparelist.index(max(comparelist[:]))
        sortedlist.append(location[max_pos])
        comparelist[max_pos] = 0
    return sortedlist

def filtered_text(text):
    ftext = ''
    firstnum = True
    for letter in text:
        if letter.isalpha():
            ftext += letter.capitalize()
        elif letter.isnumeric():
            if firstnum:
                ftext += '-'
                firstnum = False
            ftext += letter
    return ftext


def Input_Image_Processing(pathImage):
    img = cv2.imread(pathImage)
    img_original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    plt.show()
    nfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(nfilter, 30, 200)
    plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    plt.show()
    # Tìm đến các điểm chính của đường viền
    mainpoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(mainpoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    # Xác định vị trí biển số xe
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    # Làm nổi bật vị trí của biển số xe
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    # plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    # plt.show()
    (bottom_right, top_right, bottom_left, top_left) = points(location)
    # Kiểm tra độ nghiêng của biển số xe
    if top_right[0][0] - top_left[0][0] < top_right[0][1] - top_left[0][1]:
        top_width = (((bottom_left[0][0] - top_left[0][0]) ** 2) + ((bottom_left[0][1] - top_left[0][1]) ** 2))
        bottom_width = np.sqrt(((bottom_right[0][0] - top_right[0][0]) ** 2) + ((bottom_right[0][1] - top_right[0][1]) ** 2))
        right_height = np.sqrt(((bottom_right[0][0] - bottom_left[0][0]) ** 2) + ((bottom_right[0][1] - bottom_left[0][1]) ** 2))
        left_height = np.sqrt(((top_left[0][0] - top_right[0][0]) ** 2) + ((top_left[0][1] - top_right[0][1]) ** 2))
        max_width = max(int(bottom_width), int(top_width)) // 100
        max_height = max(int(right_height), int(left_height))
        # Xác định giá trị điểm (vị trí) mong muốn của ảnh đầu ra
        input_points = np.float32([top_left[0], top_right[0], bottom_left[0], bottom_right[0]])
        converted_points = np.float32([[0, 0], [0, max_height], [max_width, 0], [max_width, max_height]])
    else:
        top_width = (((top_right[0][0] - top_left[0][0]) ** 2) + ((top_right[0][1] - top_left[0][1]) ** 2))
        bottom_width = np.sqrt(((bottom_right[0][0] - bottom_left[0][0]) ** 2) + ((bottom_right[0][1] - bottom_left[0][1]) ** 2))
        right_height = np.sqrt(((top_right[0][0] - bottom_right[0][0]) ** 2) + ((top_right[0][1] - bottom_right[0][1]) ** 2))
        left_height = np.sqrt(((top_left[0][0] - bottom_left[0][0]) ** 2) + ((top_left[0][1] - bottom_left[0][1]) ** 2))
        max_width = max(int(bottom_width), int(top_width)) // 100
        max_height = max(int(right_height), int(left_height))

        input_points = np.float32([top_left[0], top_right[0], bottom_left[0], bottom_right[0]])
        converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])
    # Chuyển đổi phối cảnh
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    img_output = cv2.warpPerspective(img_original, matrix, (max_width, max_height))
    # plt.imshow(cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB))
    cv2.imwrite('output_image.jpg', img_output)
    # plt.show()

def Load_List_Char():
    img = cv2.imread('output_image.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = thresh.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    char_list = []
    for y in range(h):
        for x in range(w):
            if thresh[y][x] > 0:
                cv2.floodFill(thresh, mask, (x, y), 255, flags=cv2.FLOODFILL_MASK_ONLY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        roi = img[y:y + h, x:x + w]
        height, width, channels = roi.shape
        # print('char_{}.jpg'.format(i), 'Size:', width, 'x', height)
        if height >= 10 and width >= 10:
            roi = cv2.resize(roi, (20, 20))
            char_list.append(roi)
            # cv2.imshow('char_{}.jpg'.format(i), roi)
    return char_list

def Run():
    global inverted_list
    char_list = Load_List_Char()
    gray_list = []
    inverted_list = []
    for i, char_img in enumerate(char_list):
        # cv2.imshow('Char {}'.format(i), char_img)
        # print('Kích thước của bức ảnh thứ {} là'.format(i), char_img.shape)
        gray_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        gray_list.append(gray_img)
        # print('Kích thước của ảnh xám thứ {} là'.format(i), gray_img.shape)
        inverted_img = cv2.bitwise_not(gray_img)
        inverted_list.append(inverted_img)
        # cv2.imshow('Char {}'.format(i), inverted_img)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

def fix_dimension(img):
    new_img = np.zeros((28, 28, 3))
    for i in range(3):
        new_img[:, :, i] = img
    return new_img

def show_results():
    dic = {}
    Run()
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, c in enumerate(characters):
        dic[i] = c
    output = []
    for i, ch in enumerate(inverted_list):
        img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3)
        y_prob = model.predict(img)[0]
        y_ = np.argmax(y_prob)
        character = dic[y_]
        output.append(character)
    plate_number = ''.join(output)
    return plate_number

def S():
    Load_List_Char()
    Run()
    plt.figure(figsize=(10, 6))
    for i, ch in enumerate(inverted_list):
        img = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        plt.subplot(3, 4, i + 1)
        # plt.imshow(img, cmap='gray')
        plt.title(f'predicted: {show_results()[i]}')
        plt.axis('off')
    # plt.show()
# S()

def open_file():
    try:
        root.filename = filedialog.askopenfilename(initialdir="D:/Tu_Hoc_Lap_Trinh/License_Plate_Recognition/Image_Test/", title="Thực hiện tìm kiếm ảnh")
        PathImage = root.filename
        Upload_Image = Image.open(PathImage)
        print('Đường đẫn của bức ảnh là: ', root.filename)
        Upload_Image.thumbnail(((root.winfo_width()/1.5), (root.winfo_height()/1.5)))
        Images = ImageTk.PhotoImage(Upload_Image)
        Show_Image.configure(image=Images)
        Show_Image.image = Images
        Input_Image_Processing(PathImage)
        Run()
        sign = show_results()
        Label_Predicted_Results.configure(text=sign, foreground='#011638')
        print(sign)
    except:
        Label_Predicted_Results.configure(foreground='#011638', text='Sự cố!!!!')
        pass

my_button = tk.Button(root, text="Tải ảnh", command=open_file)
my_button.pack(side="bottom", pady=30)
Show_Image.pack(side="bottom")
Label_Predicted_Results.pack(side="bottom")
root.mainloop()