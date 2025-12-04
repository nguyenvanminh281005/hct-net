from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from .paths import Path
from .transform import custom_transforms as tr
import cv2
import tifffile as tiff

class CVCSegmentation(Dataset):
    """
    CVC dataset
    """
    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('cvc'),
                 split='train',
                 ):
        """
        :param base_dir: path to CVC dataset directory
        :param split: train/valid
        :param transform: transform to apply     CXHXW   1XHXW
        """
        super(CVCSegmentation, self).__init__()
        self.flag=split
        self.args = args
        self._base_dir = base_dir
        
        # CVC dataset structure: cvc/PNG/Original and cvc/PNG/Ground Truth
        # Use PNG by default, can change to TIF if needed
        image_format = 'PNG'  # Change to 'TIF' if you want to use TIFF format
        self._image_dir = os.path.join(self._base_dir, image_format, 'Original')
        self._cat_dir = os.path.join(self._base_dir, image_format, 'Ground Truth')
        
        # Check if directories exist
        if not os.path.exists(self._image_dir):
            raise ValueError(f"Image directory not found: {self._image_dir}")
        if not os.path.exists(self._cat_dir):
            raise ValueError(f"Ground truth directory not found: {self._cat_dir}")

        self.filenames = os.listdir(self._image_dir)
        self.data_list = []
        self.gt_list = []
        self.img_ids=[]
        
        # Sort filenames by numeric value (1.png, 2.png, etc.)
        import re
        def extract_number(filename):
            numbers = re.findall(r'\d+', filename.split('.')[0])
            return int(numbers[0]) if numbers else 0
        
        self.filenames = sorted(self.filenames, key=extract_number)
        
        # Split dataset into train/valid (80/20 split by default)
        total_images = len(self.filenames)
        train_size = int(total_images * 0.8)
        
        if split == 'train':
            self.filenames = self.filenames[:train_size]
        else:  # valid
            self.filenames = self.filenames[train_size:]
        
        for filename in self.filenames:
            index_name = filename.split(".")[0]
            img_path = os.path.join(self._image_dir, filename)
            gt_path = os.path.join(self._cat_dir, filename)
            
            # Check if both image and ground truth exist
            if os.path.exists(img_path) and os.path.exists(gt_path):
                self.data_list.append(img_path)
                self.gt_list.append(gt_path)
                self.img_ids.append(index_name)
        
        assert (len(self.data_list) == len(self.gt_list)), "Mismatch between images and ground truth"
        
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.data_list)))


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        index_name = self.img_ids[index]
        if self.flag == "train":
            image,mask= self.transform_tr(_img,_target)
            return image,mask,index_name
        else:
            image, mask=self.transform_val(_img,_target)
            #image, mask=self.transform_test(_img,_target)
            return image,mask,index_name

    def _make_img_gt_point_pair(self, index):
        # Load image - support both PNG and TIF formats
        img_path = self.data_list[index]
        if img_path.endswith('.tif'):
            _img = tiff.imread(img_path)
            _img = Image.fromarray(_img).convert("RGB")
        else:  # PNG or other formats
            _img = Image.open(img_path).convert("RGB")
        
        # Load ground truth mask
        _target = Image.open(self.gt_list[index])
        
        # Convert to grayscale if needed
        if _target.mode != 'L':
            _target = _target.convert('L')
        
        # Normalize to 0-1 range
        _target = (np.asarray(_target) / 255).astype(np.float32)
        _target = Image.fromarray(_target)
        
        return _img, _target


    def transform_tr(self,img,mask):
        composed_transforms=tr.Compose([
            #tr.RandomVerticallyFlip(),
            tr.RandomHorizontalFlip(),
            tr.RandomRotate(),
            # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.FixedResize(size=(256,256)),
            #tr.RandomColorJitter(),
            #tr.RandomGaussianBlur(),
            tr.Totensor_Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        image, mask = composed_transforms(img, mask)
        return image, mask

    def transform_val(self, img,mask):
        composed_transforms = tr.Compose([
            tr.FixedResize(size=(256, 256)),
            tr.Totensor_Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        image, mask = composed_transforms(img, mask)
        return image, mask

    def transform_test(self,img,mask):
        composed_transforms = tr.Compose([
            tr.FixedResize(size=(256,256)),  # h,w
            tr.Totensor_Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        return composed_transforms(img,mask)

    def __str__(self):
        return 'CVC(split=' + str(self.flag) + ')'


if __name__=="__main__":


    import argparse
    from torch.utils.data import DataLoader
    import torch

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset='cvc'
    dataset=CVCSegmentation(args,split="valid")
    dataloader=DataLoader(dataset,batch_size=1,num_workers=2,shuffle=True)
    count__=0
    for i,sample in enumerate(dataloader):
        images,labels,name=sample
        print(name)
        #labels = labels == torch.max(labels)
        n,c,h,w=labels.size()
        image=images.numpy()[0].transpose(1,2,0)*255
        image=image.astype(np.uint8)
        print(torch.max(labels))
        print(np.unique(labels.data.numpy()))
        # 0<x<1
        # mask=labels.data.numpy()
        # mask1=mask>0
        # mask2=mask<np.max(mask)
        # # 0~1
        # mask_0_1=mask1*mask2
        # # just 1
        # mask3=mask==np.max(mask)
        # num01=np.sum(mask_0_1)
        # num1=np.sum(mask3)
        # print("num01:{} num1:{}  num01rate:{}  num1 rate:{}".format(num01,num1,num01/(h*w),num1/(h*w)))
        #
        # mask1=mask1*255
        # mask3=mask3*255
        # mask_0_1=mask_0_1*255
        # mask1=Image.fromarray(mask1[0][0].astype(np.uint8))
        # mask3=Image.fromarray(mask3[0][0].astype(np.uint8))
        # mask_0_1=Image.fromarray(mask_0_1[0][0].astype(np.uint8))
        # mask1.show('>0')
        # mask3.show('0-1')
        # mask_0_1.show('=1')
        #

        label=labels.numpy()[0][0].astype(np.uint8)
        image=Image.fromarray(image)
        image.show("Image")
        label[label==1]=255
        label=Image.fromarray(label)
        label.show("{}".format(name))

        count__+=1
        if count__>10:
            break

