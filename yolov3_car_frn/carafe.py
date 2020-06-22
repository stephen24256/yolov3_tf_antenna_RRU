import torch
import torch.nn as nn
import torch.nn.functional as F
import time
class Carafe(nn.Module):
    def __init__(self, int_c, out_c, up_ratio=2, kernel_size=3):
        super(Carafe, self).__init__()
        self.up_ratio = up_ratio
        self.kernel_size = kernel_size
        self.fill = nn.ZeroPad2d(padding=(self.kernel_size//2, self.kernel_size//2, self.kernel_size//2, self.kernel_size//2))
        self.layer0 = nn.Sequential(
            nn.Conv2d(int_c, out_c, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(out_c, up_ratio ** 2 * kernel_size ** 2, 1, 1),
            nn.BatchNorm2d(up_ratio ** 2 * kernel_size ** 2),
            nn.ReLU(True),
        )
        self.pixelshuffle = nn.PixelShuffle(up_ratio)
        self.fill = nn.ZeroPad2d(padding=(1,1,1,1))

    def forward(self,x):
        input_x = x
        N, C, H, W = x.size()

        x = self.layer0(x)
        x = self.encoder(x)
        x = self.pixelshuffle(x)
        x = torch.softmax(x,dim=1)

        x = x.unfold(2,self.up_ratio,step=self.up_ratio)
        x = x.unfold(3,self.up_ratio,step=self.up_ratio)
        x = x.reshape(N,self.kernel_size**2,H,W,self.up_ratio**2)
        x = x.permute(0,2,3,1,4)

        input_x = self.fill(input_x)
        input_x = input_x.unfold(2,self.kernel_size,step=1)
        input_x = input_x.unfold(3,self.kernel_size,step=1)
        input_x = input_x.reshape(N,C,H,W,-1)
        input_x = input_x.permute(0,2,3,1,4)

        out = torch.matmul(input_x,x)
        out = out.reshape(N,H,W,-1)
        out = out.permute(0,3,1,2)
        out = self.pixelshuffle(out)

        return out

if __name__ == '__main__':
    x = torch.randint(1, 10, (3, 128, 196, 196), dtype=torch.float32)
    time0 = time.time()
    net = Carafe(128, 32, 2, 3)
    y = net(x)
    time1 = time.time()
    # print(x.shape)
    print(y.shape)
    print("花费时间",time1-time0)
    t1 = time.time()
    y1 = F.interpolate(x,scale_factor=2,mode="nearest")
    t2 = time.time()
    print(y1.shape)
    print("花费时间",t2-t1)
