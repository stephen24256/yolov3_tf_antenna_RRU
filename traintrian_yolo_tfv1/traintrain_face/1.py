from PIL import Image


img1 = Image.open("dco/15.png")
img2 = Image.open("dco/16.png")
img1 = img1.convert('RGB')
img2 = img2.convert('RGB')

img1 = img1.resize((64,64))
img2 = img2.resize((64,64))
img1.save('test_img1/15.jpg')
img2.save('test_img1/16.jpg')




