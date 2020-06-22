import dataset
from model import *
import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn


# 损失
def loss_fn(output, target, alpha):

    conf_loss_fn = torch.nn.BCEWithLogitsLoss()  # 标签one_hot形式
    crood_loss_fn = torch.nn.MSELoss()
    cls_loss_fn = torch.nn.CrossEntropyLoss()  # 标签索引形式
    #[N,C,H,W]-->>[N,H,W,C]
    output = output.permute(0, 2, 3, 1)  # N C H W ——》 N H W C
    # [N,C,H,W]-->>[N,H,W,3,15]
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
    output = output.cpu().double()
    mask_obj = target[..., 0] > 0  # 找到掩码
    output_obj = output[mask_obj]  #
    target_obj = target[mask_obj]
    loss_obj_conf = conf_loss_fn(output_obj[:, 0], target_obj[:, 0])
    loss_obj_crood = crood_loss_fn(output_obj[:, 1:5], target_obj[:, 1:5])
    # loss_obj_cls = cls_loss_fn(output_obj[:, 5:], target_obj[:, 5:])
    loss_obj_cls = conf_loss_fn(output_obj[:, 5:], target_obj[:, 5:])
    loss_obj = loss_obj_conf + loss_obj_crood + loss_obj_cls

    mask_noobj = target[..., 0] == 0
    output_noobj = output[mask_noobj]
    target_noobj = target[mask_noobj]
    loss_noobj = conf_loss_fn(output_noobj[:, 0], target_noobj[:, 0])
    loss = alpha * loss_obj + (1 - alpha) * loss_noobj

    return loss

if __name__ == '__main__':
    save_path = "models/net_yolo_9700.pth"
    myDataset = dataset.MyDataset()
    train_loader = DataLoader(myDataset, batch_size=3, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MainNet().to(device)

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    else:
        print("NO Param")

    net.train()
    opt = torch.optim.Adam(net.parameters())

    epoch = 11400
    loss = 0
    while True:
        for target_13, target_26, target_52, img_data in train_loader:
            img_data = img_data.to(device)
            output_13, output_26, output_52 = net(img_data)
            loss_13 = loss_fn(output_13, target_13, 0.9)
            loss_26 = loss_fn(output_26, target_26, 0.9)
            loss_52 = loss_fn(output_52, target_52, 0.9)
            loss = loss_13 + loss_26 + loss_52

            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % 5 == 0:
            print(loss.item())
            torch.save(net.state_dict(), save_path)
            print('save {}'.format(epoch))
        epoch += 1
        if epoch % 100 == 0:
            torch.save(net.state_dict(), "models/net_yolo_{}.pth".format(epoch))
