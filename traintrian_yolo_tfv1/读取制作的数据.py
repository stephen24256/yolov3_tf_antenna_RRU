import os
from xml.etree import ElementTree as ET

with open(r".\data\dataset\antenna_train1.txt","w") as F:
    root_dir = r".\data\antenna"
    dir_path = os.path.join(root_dir,"JPEGImages")
    xml_dir_path = os.path.join(root_dir,"Annotations")
    files = os.listdir(xml_dir_path)
    # print("files",files)
    files.sort(key=lambda x: int(x[0:9])+int(x.split('_')[1].split(".")[0]))
    # print("files",files)
    # exit()
    label1 = []
    ima_error_path = []
    for name in files:
        # 1. 构建路径
        xml_file = os.path.join(xml_dir_path, name)
        # 2. 构建数据读取
        tree = ET.parse(xml_file)
        # 3. 得到xml的根节点
        root = tree.getroot()
        # print("10",root)
        # 4. 获取路径信息
        path = root.find("path").text
        # print("11",path)
        filename = path.split(".")

        filename1 = filename[0].split("\\")
        filename1 = filename1[-1] + "." + filename[1]
        # print("12",filename1)
        # xml_labeled = root.find("labeled").text
        # print("xml_labeled",xml_labeled)

        # print("1+2",1+2)
        # 6. 加载目标
        outputs = root.find("outputs")
        # print("13",outputs)
        # print("***************")
        image_file_path = os.path.join(dir_path, filename1)
        object = outputs.find('object')
        if object==None:
            print("image_file_path",image_file_path)
            ima_error_path.append(image_file_path)
            continue
        # 5. 构建路径
        F.writelines(image_file_path)
        try:
            for obj in object:
                # a. 得到标签
                label = obj.find('name').text
                label1.append(label)
                # print("label",label)
                # exit()
                # b. 获取坐标信息
                bbox = obj.find("bndbox")
                xmin = bbox.find("xmin").text
                ymin = bbox.find("ymin").text
                xmax = bbox.find("xmax").text
                ymax = bbox.find("ymax").text
                #
                # print("xmin, ymin, xmax, ymax",xmin, ymin, xmax, ymax,type(xmin))
                # exit()
                # 这里输出为零的原因是：因为只有一个类别

                if label == "A0":
                    F.writelines(" {},{},{},{},0".format(xmin, ymin, xmax, ymax))
                else:
                    F.writelines(" {},{},{},{},1".format(xmin, ymin, xmax, ymax))
            # print("---------")
            # print("56lab",label1)
            F.writelines('\n')
        except:
            ima_error_path.append(image_file_path)
            print("错误")
    print('错误地址收集',ima_error_path)

