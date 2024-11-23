
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import scipy.io as sio

#several data augumentation strategies
def cv_random_flip(img, label,depth):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    #left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

    return img, label, depth
def randomCrop(image, label,depth):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region),depth.crop(random_region)
def randomRotation(image,label,depth):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image=image.rotate(random_angle, mode)
        label=label.rotate(random_angle, mode)
        depth=depth.rotate(random_angle, mode)
    return image,label,depth
def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
#实现MixUp和CutMix数据增强
def mixup(image1, image2, label1, label2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    image = lam * np.array(image1) + (1 - lam) * np.array(image2)
    label = lam * np.array(label1) + (1 - lam) * np.array(label2)
    return Image.fromarray(image.astype(np.uint8)), Image.fromarray(label.astype(np.uint8))

def cutmix(image1, image2, label1, label2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    width, height = image1.size
    cut_w = np.random.randint(0, width)
    cut_h = np.random.randint(0, height)

    # Cut a patch from image2
    patch = image2.crop((cut_w, cut_h, cut_w + width, cut_h + height))

    # Replace part of image1 with the patch from image2
    image1.paste(patch, (cut_w, cut_h))

    # For label, just replace the same region
    label1.paste(label2.crop((cut_w, cut_h, cut_w + width, cut_h + height)), (cut_w, cut_h))

    return image1, label1

#引入椒盐噪声和泊松噪声
def add_salt_pepper_noise(image, amount=0.01):
    img = np.array(image)
    total_pixels = img.shape[0] * img.shape[1]
    num_salt = int(amount * total_pixels)
    num_pepper = int(amount * total_pixels)

    # Salt noise
    salt_coords = [np.random.randint(0, i-1, num_salt) for i in img.shape]
    img[salt_coords[0], salt_coords[1]] = 255

    # Pepper noise
    pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in img.shape]
    img[pepper_coords[0], pepper_coords[1]] = 0

    return Image.fromarray(img)

def add_poisson_noise(image):
    img = np.array(image) / 255.0  # Normalize to [0, 1]
    noise = np.random.poisson(img * 255.0)  # Poisson noise
    noisy_img = np.clip(noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

#
def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))
def randomPeper(img):

    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):

        randX=random.randint(0,img.shape[0]-1)

        randY=random.randint(0,img.shape[1]-1)

        if random.randint(0,1)==0:

            img[randX,randY]=0

        else:

            img[randX,randY]=255
    return Image.fromarray(img)

# dataset for training
#The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
#(e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
# class SalObjDataset(data.Dataset):
#     def __init__(self, image_root, gt_root,depth_root, trainsize):
#         self.trainsize = trainsize
#         file_names_1 = os.listdir(image_root)
#         # file_names= file_names_1+ file_names_2
#
#         self.images = []
#         self.gts = []
#         self.depths= []
#
#         for i, name in enumerate(file_names_1):
#             if not name.endswith('.jpg'):
#                 continue
#             self.gts.append(
#                 os.path.join(gt_root, name[:-4]+'.png')
#             )
#
#             self.images.append(
#                 os.path.join(image_root, name)
#             )
#             self.depths.append(
#                 os.path.join(depth_root, name[:-4]+'.jpg')
#             )
#
#         self.size = len(self.images)
#         self.img_transform = transforms.Compose([
#             transforms.Resize((self.trainsize, self.trainsize)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.525, 0.590, 0.537], [0.177, 0.167, 0.176]),
#             ])
#         self.gt_transform = transforms.Compose([
#             transforms.Resize((self.trainsize, self.trainsize)),
#             transforms.ToTensor()])
#         self.depths_transform = transforms.Compose([
#             transforms.Resize((self.trainsize, self.trainsize)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.736, 0.346, 0.339], [0.179, 0.196, 0.169]),
#             ])
#
#     def add_random_noise(image, noise_level=0.1):
#         img_array = np.array(image)
#         noise = np.random.randn(*img_array.shape) * noise_level * 255  # 乘以255以适应图像的像素范围
#         noisy_image = np.clip(img_array + noise, 0, 255).astype(np.uint8)
#         return Image.fromarray(noisy_image)
#
#     def __getitem__(self, index):
#         image = self.rgb_loader(self.images[index])
#         gt = self.binary_loader(self.gts[index])
#         depth=self.rgb_loader(self.depths[index])
#
#         image,gt,depth =cv_random_flip(image,gt,depth)
#         image,gt,depth=randomRotation(image,gt,depth)
#         image=colorEnhance(image)
#         depth=colorEnhance(depth)
#
#         image = self.add_random_noise(image, noise_level=0.1)
#
#         gt=randomPeper(gt)
#
#         image = self.img_transform(image)
#         gt = self.gt_transform(gt)
#         depth=self.depths_transform(depth)
#
#         return image, gt, depth
#
#     def filter_files(self):
#         assert len(self.images) == len(self.gts) and len(self.gts)==len(self.images)
#         images = []
#         gts = []
#         depths=[]
#         for img_path, gt_path,depth_path in zip(self.images, self.gts, self.depths):
#             img = Image.open(img_path)
#             gt = Image.open(gt_path)
#             depth= sio.loadmat(depth_path, verify_compressed_data_integrity=False)
#             if img.size == gt.size and gt.size==depth.size:
#                 images.append(img_path)
#                 gts.append(gt_path)
#                 depths.append(depth_path)
#         self.images = images
#         self.gts = gts
#         self.depths=depths
#
#     def rgb_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('RGB')
#
#     def binary_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('L')
#
#     def resize(self, img, gt, depth):
#         assert img.size == gt.size and gt.size==depth.size
#         w, h = img.size
#         if h < self.trainsize or w < self.trainsize:
#             h = max(h, self.trainsize)
#             w = max(w, self.trainsize)
#             return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST),depth.resize((w, h), Image.NEAREST)
#         else:
#             return img, gt, depth
#
#     def __len__(self):
#         return self.size

class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, trainsize, noise_level=0.1, use_mixup=False, use_cutmix=False):
        self.trainsize = trainsize
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        file_names_1 = os.listdir(image_root)
        self.images = []
        self.gts = []
        self.depths = []

        for i, name in enumerate(file_names_1):
            if not name.endswith('.jpg'):
                continue
            self.gts.append(os.path.join(gt_root, name[:-4] + '.png'))
            self.images.append(os.path.join(image_root, name))
            self.depths.append(os.path.join(depth_root, name[:-4] + '.jpg'))

        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.525, 0.590, 0.537], [0.177, 0.167, 0.176]),
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.736, 0.346, 0.339], [0.179, 0.196, 0.169]),
        ])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.rgb_loader(self.depths[index])

        # Data augmentation
        image, gt, depth = cv_random_flip(image, gt, depth)
        image, gt, depth = randomRotation(image, gt, depth)
        image = colorEnhance(image)
        depth = colorEnhance(depth)

        # Add noise
        image = self.add_random_noise(image, noise_level=0.1)
        gt = randomPeper(gt)
        image = add_salt_pepper_noise(image, amount=0.01)
        image = add_poisson_noise(image)

        # Optionally apply MixUp or CutMix
        if self.use_mixup and random.random() > 0.5:
            idx2 = random.randint(0, len(self.images) - 1)
            image2 = self.rgb_loader(self.images[idx2])
            gt2 = self.binary_loader(self.gts[idx2])
            image, gt = mixup(image, image2, gt, gt2)

        if self.use_cutmix and random.random() > 0.5:
            idx2 = random.randint(0, len(self.images) - 1)
            image2 = self.rgb_loader(self.images[idx2])
            gt2 = self.binary_loader(self.gts[idx2])
            image, gt = cutmix(image, image2, gt, gt2)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)

        return image, gt, depth

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def add_random_noise(self, image, noise_level=0.1):
        img_array = np.array(image)
        noise = np.random.randn(*img_array.shape) * noise_level * 255  # 乘以255以适应图像的像素范围
        noisy_image = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)
    def __len__(self):
        return self.size
    
#dataloader for training
# def get_loader(image_root, gt_root,depth_root, batchsize, trainsize=384, shuffle=True, num_workers=8, pin_memory=True):
#
#     dataset = SalObjDataset(image_root, gt_root, depth_root,trainsize)
#     data_loader = data.DataLoader(dataset=dataset,
#                                   batch_size=batchsize,
#                                   shuffle=shuffle,
#                                   num_workers=num_workers,
#                                   pin_memory=pin_memory,
#                                 #   drop_last= True)
#                                                     )
#     return data_loader
def get_loader(image_root, gt_root, depth_root, batchsize, trainsize=384, shuffle=True, num_workers=8, pin_memory=True, use_mixup=False, use_cutmix=False):
    dataset = SalObjDataset(image_root, gt_root, depth_root, trainsize, use_mixup=use_mixup, use_cutmix=use_cutmix)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

# test dataset and loader
class Test_dataset(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, testsize):
        super().__init__()
        self.testsize = testsize

        file_names = os.listdir(image_root)
        self.images = []
        self.gts = []
        self.depths = []

        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.gts.append(
                os.path.join(gt_root, name[:-4] + '.png')
            )

            self.images.append(
                os.path.join(image_root, name)
            )
            self.depths.append(
                os.path.join(depth_root, name[:-4] + '.jpg')
            )
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.525, 0.590, 0.537], [0.177, 0.167, 0.176]),
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)),
             transforms.ToTensor(),
             transforms.Normalize([0.736, 0.346, 0.339], [0.179, 0.196, 0.169]),
             ])

        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.rgb_loader(self.depths[index])

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        # h*w
        gt_shape = gt.shape
        depth = self.depths_transform(depth)

        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        return image, gt_shape, depth, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def mat_loader(self, path):
        depth = sio.loadmat(path, verify_compressed_data_integrity=False)
        depth = depth['img']
        return depth

    def __len__(self):
        return self.size


class Test_dataset2(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, testsize):
        super().__init__()
        self.testsize = testsize

        file_names = os.listdir(image_root)
        self.images = []
        self.gts = []
        self.depths = []

        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.gts.append(
                os.path.join(gt_root, name[:-4] + '.jpg')
            )

            self.images.append(
                os.path.join(image_root, name)
            )
            self.depths.append(
                os.path.join(depth_root, name[:-4] + '.jpg')
            )
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.525, 0.590, 0.537], [0.177, 0.167, 0.176])
        ])
        self.gt_transform = transforms.Compose([
            # transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)),
             transforms.ToTensor(),
             transforms.Normalize([0.736, 0.346, 0.339], [0.179, 0.196, 0.169]),
             ])

        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.rgb_loader(self.depths[index])

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        # h*w
        gt_shape = gt.shape
        depth = self.depths_transform(depth)

        name = self.images[index].split('/')[-1]
        # if name.endswith('.jpg'):
        #     name = name.split('.jpg')[0] + '.png'

        return image, gt_shape, depth, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def mat_loader(self, path):
        depth = sio.loadmat(path, verify_compressed_data_integrity=False)
        depth = depth['img']
        # depth = np.array(depth, dtype=np.int32)
        return depth

    def __len__(self):
        return self.size
