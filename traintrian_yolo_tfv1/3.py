import cv2
import random
aa = ['000223195_9.xml', '000236408_4.xml', '000236408_5.xml', '000236408_6.xml', '000236408_7.xml', '000223195_10.xml', '000223195_11.xml', '000223195_12.xml']
aa.sort(key=lambda x: int(x[0:9])+int(x.split('_')[1].split(".")[0]))

print(aa)

bb = ['0002231959.xml', '0002364084.xml', '0002364085.xml', '0002364086.xml', '0002364087.xml', '00022319510.xml', '00022319511.xml', '00022319512.xml']
bb.sort(key=lambda x: x.split("."))
print(bb)

a = random.randint(-2,2)
print(a)