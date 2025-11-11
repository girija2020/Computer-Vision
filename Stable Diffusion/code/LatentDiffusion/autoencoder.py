import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
import cv2
import os
from tqdm import tqdm

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(2,1),
            nn.Conv2d(64, 16, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=5, padding=4),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        E = self.encoder(img)
        D = self.decoder(E)
        return D

class FeatureLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = nn.Sequential(*list(model.children())[:-1])
    
    def forward(self, imgs, decoded_imgs):
        loss = self.model(imgs) - self.model(decoded_imgs)
        return loss.mean()**2
    
class FaceData(Dataset):
    def __init__(self, filepath, transform):
        self.img_paths = []
        for folder in os.listdir(filepath):
            for img_file in os.listdir(os.path.join(filepath, folder)):
                self.img_paths.append(os.path.join(filepath, folder, img_file))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img)

    
transform = transforms.Compose(
    [
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomCrop((112,112)),
    transforms.RandomHorizontalFlip()
    ]
)


dataset = FaceData('/projectnb/cs585bp/students/dlgirija/colorferet/images_extracted', transform)
train,val = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
train_dataloader = DataLoader(train, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val, batch_size=32, shuffle=True)
lossModel = resnet50(ResNet50_Weights)#.to('cuda')
lossModel.eval()
loss = FeatureLoss(lossModel).to('cuda')
model = Autoencoder().to('cuda')
optimizer = optim.Adam(model.parameters(), lr=1e-3)


epochs = 10
transform_output = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
for num in range(epochs):
    nums = 0
    L = 0
    for imgs in tqdm(train_dataloader):
        imgs = imgs.to('cuda')
        decoded_imgs = model(imgs)
        optimizer.zero_grad()
        l = loss(transform_output(imgs), decoded_imgs)
        optimizer.step()
        nums += 1
        L += l
    print(L/nums)
        
# import torch
# import torch.nn as nn
# from torchvision.models import resnet50
# from torchvision.models import ResNet50_Weights
# from torchvision import transforms
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset, random_split
# import cv2
# import os
# from torch.nn.parallel import DistributedDataParallel as DDP

# # without sigmoid loss - 0.109 - the man is visible loss.mean * 2

# class Autoencoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=6, padding=3, stride=2), # 57x57
#             nn.ReLU(),
#             nn.MaxPool2d(2,1), # 56x56
#             nn.Conv2d(64, 128, kernel_size=4, padding=2, stride=2), # 29x29
#             nn.ReLU(),
#             nn.MaxPool2d(2,1), # 28x28
#             nn.Conv2d(128, 256, kernel_size=2, padding=1, stride=2), # 15x15
#             nn.ReLU(),
#             nn.MaxPool2d(2,1), # 14x14
#             nn.Conv2d(256, 512, kernel_size=2, padding=1, stride=2), # 8x8
#             nn.ReLU(),
#             nn.MaxPool2d(2,1) # 7x7
#         )
#         self.convLayer = nn.Conv2d(512, 1024, kernel_size=7, padding=0, stride=1) # 1x1
#         self.relu = nn.ReLU()
#         # [(input-1)stride]+kernel_size-2*padding_of_output
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(1024,512, kernel_size=7, stride=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(512,256, kernel_size=2, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(256,128, kernel_size=2, stride = 2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128,64, kernel_size=2, stride = 2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 3, kernel_size=2, stride = 2),
#             # nn.Sigmoid()
#         )
        
#     def forward(self, img):
#         E = self.encoder(img)
#         E = self.convLayer(E)
#         E = self.relu(E)
#         D = self.decoder(E)
#         # # print(E.shape)
#         # batch_size, C,H,W = E.shape
#         # fc1 = self.fc1(E.view(batch_size, -1))
#         # # print(fc1.shape)
#         # fc2 = self.fc2(fc1).view(batch_size, C,H,W)
#         # D = self.decoder(fc2)
#         # print(D.shape)
#         return D

# class FeatureLoss(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = nn.Sequential(*list(model.children())[:-1])
#         self.model.eval()
    
#     def forward(self, imgs, decoded_imgs):
#         with torch.no_grad():
#             loss = self.model(imgs) - self.model(decoded_imgs)
#             return loss.mean()**2 
#             # return nn.MSELoss(self.model(imgs), self.model(decoded_imgs)) #+0.0001*(imgs-decoded_imgs).mean()**2