import torch
from PIL import Image
from PIL import ImageDraw,ImageFont
import numpy as np
from tool import utils
import netsBNC
from torchvision import transforms
import time


class Detector:

    def __init__(self, pnet_param="./param/p_net.pth", rnet_param="./param/r_net.pth", onet_param="./param/o_net.pth",
                 isCuda=False):

        self.isCuda = isCuda

        self.pnet = netsBNC.PNet()
        self.rnet = netsBNC.RNet()
        self.onet = netsBNC.ONet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()
        #
        self.pnet.load_state_dict(torch.load(pnet_param,map_location='cpu'))
        self.rnet.load_state_dict(torch.load(rnet_param,map_location='cpu'))
        self.onet.load_state_dict(torch.load(onet_param,map_location='cpu'))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.__image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.4546, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def detect(self, image):
        start_time = time.time()
        pnet_boxes = self.__pnet_detect(image)
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_pnet = end_time - start_time
        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time
        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_onet = end_time - start_time
        t_sum = t_pnet + t_rnet + t_onet
        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return onet_boxes

    def __pnet_detect(self, image):
        boxes = []
        img = image
        w, h = img.size
        min_side_len = min(w, h)
        scale = 1
        while min_side_len > 12:
            img_data = self.__image_transform(img)
            if self.isCuda:
                img_data = img_data.cuda()
            img_data.unsqueeze_(0)
            _cls, _offest = self.pnet(img_data)
            cls, offest = _cls[0][0].cpu().data, _offest[0].cpu().data
            idxs = torch.nonzero(torch.gt(cls, 0.65))
            for idx in idxs:
                boxes.append(self.__box(idx, offest, cls[idx[0], idx[1]], scale))
            scale *= 0.7
            _w = int(w * scale)
            _h = int(h * scale)
            img = img.resize((_w, _h))
            min_side_len = np.minimum(_w, _h)
        return utils.nms(np.array(boxes), 0.3)

    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):

        _x1 = int(start_index[1] * stride) / scale#宽，W，x
        _y1 = int(start_index[0] * stride) / scale#高，H,y
        _x2 = int(start_index[1] * stride + side_len) / scale
        _y2 = int(start_index[0] * stride + side_len) / scale

        ow = _x2 - _x1#12
        oh = _y2 - _y1#12
        _offset = offset[:, start_index[0], start_index[1]]
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]

    def __rnet_detect(self, image, pnet_boxes):

        _img_dataset = []
        _pnet_boxes = utils.convert_to_square(pnet_boxes)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset =torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.rnet(img_dataset)

        _cls = _cls.cpu().data.numpy()

        offset = _offset.cpu().data.numpy()
        idxs, _ = np.where(_cls > 0.6)
        _box = _pnet_boxes[idxs]
        # print("_box",_box)
        _x1 = _box[:,0]
        _y1 = _box[:,1]
        _x2 = _box[:,2]
        _y2 = _box[:,3]
        ow = _x2-_x1
        oh = _y2-_y1

        x1 = _x1 + ow * offset[idxs][:,0]
        y1 = _y1 + oh * offset[idxs][:,1]
        x2 = _x2 + ow * offset[idxs][:,2]
        y2 = _y2 + oh * offset[idxs][:,3]
        cls = _cls[idxs][:,0]
        boxes= np.stack((x1,y1,x2,y2,cls),axis=1)
        return utils.nms(boxes, 0.3)

    def __onet_detect(self, image, rnet_boxes):

        _img_dataset = []
        _rnet_boxes = utils.convert_to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.onet(img_dataset)

        _cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()
        idxs, _ = np.where(_cls > 0.98)
        _box = _rnet_boxes[idxs]
        _x1 = _box[:, 0]
        _y1 = _box[:, 1]
        _x2 = _box[:, 2]
        _y2 = _box[:, 3]
        ow = _x2 - _x1
        oh = _y2 - _y1

        x1 = _x1 + ow * offset[idxs][:, 0]
        y1 = _y1 + oh * offset[idxs][:, 1]
        x2 = _x2 + ow * offset[idxs][:, 2]
        y2 = _y2 + oh * offset[idxs][:, 3]
        cls = _cls[idxs][:, 0]
        boxes = np.stack((x1, y1, x2, y2, cls), axis=1)

        return utils.nms(boxes, 0.2, isMin=True)


if __name__ == '__main__':
    x = time.time()
    with torch.no_grad() as grad:
        import os
        import numpy as np
        path = r"C:\Users\admin\Documents\Tencent Files\532653492\FileRecv\CASIA-WebFace"
        img_path = os.listdir(path)
        # print(img_path)
        path2 = []
        output_size = 182
        for i in img_path:
            path1 = os.path.join(path, i)
            # print(path1)
            path2.append(path1)
        print("path2", path2)

        # print(path2)
        m = 0
        save_path1 = r"./data/casia_maxpy_mtcnnpy_182"
        file_box = os.path.join(save_path1, "bounding_boxes.txt")
        with open(file_box, "w") as f:
            for m,j in enumerate(path2):
                print("j", j)
                path3 = os.listdir(j)
                print("path3", path3)
                kl = 0
                num_path = path2[m][-1:-8:-1][::-1]
                print("num_path", num_path)
                for n,k in enumerate(path3):
                    image_path = os.path.join(j, k)
                    # print("image_path",image_path)
                    detector = Detector()
                    with Image.open(image_path) as im:
                        # print("im",im)

                        im1 = np.array(im)
                        if im1.ndim==2:
                            continue
                        print("有多少", im1.shape)
                        boxes = detector.detect(im)
                        # print("202",boxes)
                        print("有多少个框", boxes.shape[0])
                        imDraw = ImageDraw.Draw(im)

                        for box in boxes:
                            x1 = int(box[0])
                            y1 = int(box[1])
                            x2 = int(box[2])
                            y2 = int(box[3])

                            cls = box[4]

                            print(x1, y1, x2, y2)
                            dw = x2 - x1
                            dh =  y2 - y1
                            cx = int(x1+dw/2 +1)
                            cy = int(y1+dh/2 +1)
                            x11 = int(cx - output_size/2)
                            x22 = x11 + output_size
                            y11 = int(cy - output_size/2)
                            y22 = y11 + output_size
                            print(x11, y11, x22, y22)

                            # imDraw.rectangle((x1, y1, x2, y2), outline='red')
                            # boxes_name = "face: %.2f" % cls
                            # font = ImageFont.truetype(font=r'C:\Windows\Fonts\ARLRDBD.TTF', size=25)
                            # imDraw.text((x1, y1 - 25), text=boxes_name, fill=(0, 0, 255), font=font)
                            # imDraw.rectangle((x11, y11, x22, y22), outline="blue")
                            crop_box = [x11, y11, x22, y22]
                            img_crop = im.crop(crop_box)


                            # exit()
                            save_path = os.path.join(save_path1,num_path)
                            print("save_path",save_path)
                            if not os.path.exists(save_path):
                                os.mkdir(save_path)
                            save_path = os.path.join(save_path,path3[n])
                            print("save_path",save_path)
                            # exit()
                            img_crop.save(save_path)
                            # print(boxes_name)
                            f.write('%s %d %d %d %d\n' % (save_path,x11,y11,x22,y22))

                        y = time.time()
                        print(y - x)
                        # im.show()
                        # exit()
                        # kl +=1
                        # if kl== 1:
                        #     break
                m +=1
                if m ==1:
                    break