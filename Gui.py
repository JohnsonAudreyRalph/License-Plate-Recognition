from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import Pretreatment

# Tạo cửa sổ chính
root = Tk()
root.title("Phần mềm nhận diện biển số xe ô tô")
root.geometry("850x550")

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
    file_path = filedialog.askopenfilename(initialdir="D:/Tu_Hoc_Lap_Trinh/Python/License Plate Recognition/Data/Image", title="Thực hiện tìm kiếm ảnh")
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
root.mainloop()