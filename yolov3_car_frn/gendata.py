import os
from xml.dom.minidom import parse
from PIL import Image
import numpy as np
from xml.dom.minidom import parse


class GenData:
    def __init__(self):
        self.text_path_dianli = r"antenna_train.txt"
        self.label_save_path = r"./data/person_label.txt"
        self.img_save_path = r"./data/images/"

    def gendata(self, stop_num):
        f_dianli = open(self.text_path_dianli)
        f_label = open(self.label_save_path, "w")
        img_list_dianli = f_dianli.readlines()
        # print(type(img_list_dianli))
        count = 0

        for strs in img_list_dianli:
            strs = strs.split()
            print("strs",strs)
            print("24",strs[0][26: -4])
            print("25",strs[0])
            img_name = strs[0]
            img = Image.open(img_name)
            w, h = img.size
            larger_lenth = max(h, w)
            rate = larger_lenth / 416
            img = img.resize((int(w/rate), int(h/rate)))
            img1 = Image.new("RGB", (416, 416), (0,0,0))
            img1.paste(img)
            img1.save(os.path.join(self.img_save_path, "{}.jpg".format(strs[0][26: -4])))
            f_label.write("{}.jpg ".format(strs[0][26: -4]))
            for box in strs[1:]:
                # print(box)
                box = box.split(",")
                # print(box)
                x1 = int(box[0]) / rate
                y1 = int(box[1]) / rate
                x2 = int(box[2]) / rate
                y2 = int(box[3]) / rate
                cls = box[4]
                cx = (x1+x2) / 2
                cy = (y1+y2) / 2
                w = x2 - x1
                h = y2 - y1
                f_label.write("{} {} {} {} {} ".format(cls, cx, cy, w, h))
            f_label.write("\n")
            count += 1
            if count > stop_num:
                break


if __name__ == '__main__':
    gen_data = GenData()
    gen_data.gendata(122)







