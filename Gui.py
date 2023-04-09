import os
import sys
from tkinter import *
from tkinter import filedialog

import cv2
from PIL import Image, ImageTk
import Pretreatment

# Tạo cửa sổ chính
root = Tk()
root.title("Phần mềm nhận diện biển số xe ô tô")
root.geometry("1020x550")


def Info():
    print("Đã nhận được nút thông tin tác giả")


def Change_Camera():
    print("Nhận thấy nút thay đổi camera được nhấn")


def Choose_a_Picture():
    cv2.destroyAllWindows()
    print("Đã chọn ảnh")
    # hide the widgets from the first page
    for widget in root.winfo_children():
        widget.pack_forget()
    # Tạo khung hình chứa ảnh
    canvas = Canvas(root, width=400, height=400)
    canvas.place(relx=0.25, rely=0.5, anchor=CENTER)
    image_canvas = canvas.create_image(0, 0, anchor=NW)

    # Tạo khung hình chứa ảnh sau khi xử lý
    processed_canvas = Canvas(root, width=400, height=200)
    processed_canvas.place(relx=0.75, rely=0.35, anchor=CENTER)
    processed_image_canvas = processed_canvas.create_image(0, 0, anchor=NW)

    # Tạo nút chọn ảnh, xử lý sự kiện khi nút chọn ảnh được nhấn
    def Choose_Photo():
        file_path = filedialog.askopenfilename(initialdir="D:/Tu_Hoc_Lap_Trinh/Python/License Plate Recognition/Data/Image",
                                               title="Thực hiện tìm kiếm ảnh")
        if file_path != "":
            # Hiển thị ảnh lên trên khung hình
            image = Image.open(file_path)
            print(file_path)
            image = image.resize((400, 400), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            canvas.itemconfigure(image_canvas, image=photo)
            canvas.image = photo
            Pretreatment.Pretreatment(file_path)
            # Xử lý ảnh và hiển thị kết quả
            processed_image = Image.open('Find_the_License_Plate.png')
            processed_image = processed_image.resize((400, 200), Image.ANTIALIAS)
            processed_photo = ImageTk.PhotoImage(processed_image)
            processed_canvas.itemconfigure(processed_image_canvas, image=processed_photo)
            processed_canvas.image = processed_photo
            # Hiển thị đường dẫn của ảnh
            text = Pretreatment.red_Image()
            path_label.config(text=text)
    # Tạo nút "Chọn ảnh"
    upload_button = Button(root, text="Chọn ảnh", command=Choose_Photo)
    upload_button.place(relx=0.5, rely=0.95, anchor=CENTER)
    # Tạo frame và label hiển thị đường dẫn của ảnh
    path_frame = Frame(root, width=300, height=50, bd=2, relief=SOLID, bg='black')
    path_frame.place(relx=0.75, rely=0.75, anchor=CENTER)
    path_label_name = Label(path_frame, text="Biển số tìm được: ", bg='black', fg='white')
    path_label_name.pack(side=LEFT)
    path_label = Label(path_frame, text="", width=30, height=2, bg='white', fg='black')
    path_label.pack(side=LEFT, fill=BOTH, expand=YES)


# def getCamera():
#     cap = cv2.VideoCapture(0)
#
#     def update_size():
#         global width, height
#         width = root.winfo_width()
#         height = root.winfo_height()
#         root.after(100, update_size)
#
#     label = Label(root)
#     label.pack(padx=10, pady=10)
#
#     def show_frame():
#         update_size()
#         _, frame = cap.read()
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = cv2.resize(frame, (width, height))
#         image = Image.fromarray(frame)
#         photo = ImageTk.PhotoImage(image)
#         label.config(image=photo)
#         label.image = photo
#         root.after(10, show_frame)
#     show_frame()

def getCamera():
    cap = cv2.VideoCapture(1)
    plateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
    minArea = 500

    def update_size():
        global width, height
        width = root.winfo_width()
        height = root.winfo_height()
        root.after(100, update_size)

    update_size()
    cap.set(3, width)
    cap.set(4, height)
    cap.set(10, 150)

    label = Label(root)
    label.pack(padx=10, pady=10)

    def show_frame():
        update_size()
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        numberPlates = plateCascade.detectMultiScale(frame, 1.1, 4)
        for (x, y, w, h) in numberPlates:
            area = w * h
            if area > minArea:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # cv2.putText(frame, "NumberPlate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                imgRoi = frame[y:y + h, x:x + w]
                cv2.imshow("ROI", imgRoi)
                cv2.imwrite('ROI.png', imgRoi)
        # cv2.imshow("Result", frame)
        text = Pretreatment.red_CAMERA()
        print(text)
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        frame = cv2.resize(frame, (width, height))
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo
        root.after(10, show_frame)
    show_frame()


def Choose_a_Camera():
    # hide the widgets from the main window
    for widget in root.winfo_children():
        if widget.winfo_class() != 'Menu':
            widget.pack_forget()

    # create a new frame to hold the new widgets
    camera_frame = Frame(root)
    camera_frame.pack(fill=BOTH, expand=True)
    print("Đã chọn camera")
    getCamera()


# Tạo menu
menuBar = Menu(root)
imageImage = PhotoImage(file="Image/image.png")
cameraImage = PhotoImage(file="Image/camera.png")
exitImage = PhotoImage(file="Image/exit.png")
changImage = PhotoImage(file="Image/change.png")

file_menu = Menu(menuBar, tearoff=0, font=("MV Boli", 15))
menuBar.add_cascade(label="System", menu=file_menu)
file_menu.add_command(label="Ảnh", command=Choose_a_Picture, image=imageImage, compound="left")
file_menu.add_command(label="Camera", command=Choose_a_Camera, image=cameraImage, compound="left")
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit, image=exitImage, compound="left")

edit_menu = Menu(menuBar, tearoff=0)
menuBar.add_cascade(label="Cài đặt", menu=edit_menu)
edit_menu.add_command(label="Thay đổi camera", image=changImage, compound="left", command=Change_Camera)
# edit_menu.add_command(label="Copy")
# edit_menu.add_command(label="Paste")

infoMenu = Menu(menuBar, tearoff=0)
menuBar.add_cascade(label="Thông tin", menu=infoMenu)
infoMenu.add_command(label="Tác giả", command=Info)

# Thêm menu vào cửa sổ chính
root.config(menu=menuBar)
root.mainloop()
