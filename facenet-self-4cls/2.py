import os
import numpy as np
import json

# with open("feature1.txt", "r") as f:
#     a = f.readlines()
#     print(a)

# b = []
# a = np.arange(12).reshape(3,4)
# # c = np.array([[0],[1],[2]])
# c = np.array([0,1,2])
# c = c[:,None]
# print(c)
# for i in range(a.shape[0]):
#     # print(a[i],c[i])
#     d = np.concatenate((a[i],c[i]))
#     b.append(d)
# print(b)
# b = np.array(b)
# print(b)

def encoding_Facestr(image_np):
    encoding_arrar_list = image_np.tolist()
    encoding_list_str = [str(i) for i in encoding_arrar_list]
    encoding_str = ','.join(encoding_list_str)
    return encoding_str

def decoding_facestr(image_str):
    dlist = image_str.strip(' ').split(',')
    dfloat = list(map(float,dlist))
    face_encoding = np.array(dfloat)
    return face_encoding


image_np = np.random.randn(3,4)
print(image_np)
st = encoding_Facestr(image_np)
print(st,type(st))
gt = '[1 2 3 4]'
dlist = gt.strip(' ').split(',')
print(dlist)
# dt = decoding_facestr(gt)
# print(dt)

dlist = '["1","2"]'
dfloat = list(map(float,dlist))
print(dfloat)
