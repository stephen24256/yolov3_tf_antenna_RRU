import os
import h5py
import numpy as np
import torch
# torch.set_printoptions(edgeitems=200,linewidth=200)
# np.set_printoptions(edgeitems=200,linewidth=200,suppress=True)

a = np.arange(12).reshape(3,4)
# print(a)

b = {}
for i,j in enumerate(a):
    b[str(j)]=i
print(b,len(b))
print("=========")

for key,vule in b.items():
    # print("key",key,type(key))
    a2 = np.array(key)
    # print("np",a2,type(a2))
    # print(vule,type(vule))
c =np.array([0,1,2,3])
print("c",c)
print(b[str(c)])
print(b.keys(),b.values())

for m in b.keys():
    print(m,type(m))
    c1 = np.array(m)
    print(c1)

exit()

# dict  字典
dict1={"name":"撒贝宁","sex":"男","age":40,'age':30} # 如果出现同名key,后面的会覆盖前面
print(dict1)
print(dict1['sex'])
print(dict1[0])


#修改
# dict1['age']=40

# 新增
# dict1['role']="央视主持人"
# print(dict1)


# 删除
# del dict1['age']
# 清空 clear
# dict1.clear()
#
# print(type(dict1))


# 函数
# str() 强制转换成字符串
# print(dict1)
# print(type(str(dict1)))
# dict1_str=str(dict1)

# fromkeys() 把一个序列中的值所谓字典中的key,生成一个字典,可以赋初始值
# 有一个有关人名的列表,想要把列表中的每个值作为key,然后赋值
names=['Tom','Jerry',"Anni"]
infos={}
infos[names[0]]={}
infos[names[1]]={}
infos[names[2]]={}
print(infos)
infos=dict.fromkeys(names)
print(infos)
print(infos.get('Tom','1111'))


# 取值 get('key'[,default]) 从字典中取值,如果该key存在,返回值,如果不存在,返回设定的默认值
# print(dict1['name']) # 取一个存在的值,ok的
# print(dict1['role']) # 如果取了一个不存在的值,会报错
# if 'role' in dict1:
#     print(dict1['role'])
# else:
#     print("没有此属性")

# print(dict1.get("role",'没有此属性'))
# # print(dict1.get("name",'没有此属性'))

# setdefault(key[,default]) 取值 如果没有key 的时候,要比get多一个 新增操作
# print(dict1.setdefault("name"))
# print(dict1.setdefault('role',"没有此属性"))
# print(dict1)


# keys()  返回字典中所有的key , values()  返回所有的值
# print(list(dict1.keys())[0])
# print(dict1.values())


#
# favorite_places={}
# name=input("请输入人名:")
# place=input("请输入喜欢的地方(多个地方使用逗号隔开):")
# favorite_places[name]=place.split(",")
# name=input("请输入人名:")
# place=input("请输入喜欢的地方(多个地方使用逗号隔开):")
# favorite_places[name]=place.split(",")
# name=input("请输入人名:")
# place=input("请输入喜欢的地方(多个地方使用逗号隔开):")
# favorite_places[name]=place.split(",")
# print(favorite_places)

# favorite_places={'刘青': ['北京', '上海', '广州'], '刘兰': ['北京', '南京', '苏州'], '刘黄': ['深圳', '杭州', '新疆']}
# where=input("请输入一个地址:")
# names=list(favorite_places.keys())
# places=list(favorite_places.values())
# print(names)
# print("%s喜欢这个地方吗?"%(names[0]),where in places[0])
# print("%s喜欢这个地方吗?"%(names[1]),where in places[1])
# print("%s喜欢这个地方吗?"%(names[2]),where in places[2])
# i=0
# while i<len(favorite_places):
#     if where in places[i]:
#         print("%s喜欢这个地方"%(names[i]))
#     i+=1

# haha={"type":"哈士奇","master":"小王"}
# xiaohuang={"type":"金毛",'master':"小李"}
# xiaohei={"type":"德牧",'master':"小黑"}
# pets=[]
# pets.append(haha)
# pets.append(xiaohei)
# pets.append(xiaohuang)
# print(pets)
#
# print("我是%s,我的狗的类型是%s"%(pets[0]['master'],pets[0]['type']))
# print("我是%s,我的狗的类型是%s"%(pets[1]['master'],pets[1]['type']))
# print("我是%s,我的狗的类型是%s"%(pets[2]['master'],pets[2]['type']))


