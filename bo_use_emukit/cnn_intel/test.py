from torchvision import models, datasets, transforms
import torch.utils.data
from sklearn.model_selection import train_test_split

transforms = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor(),
                                       ]) # could be augmentation

train_dir = '../data/intel_image/seg_train/seg_train'
test_dir = '../data/intel_image/seg_test/seg_test'

train_data = datasets.ImageFolder(train_dir, transform=transforms)
test_data = datasets.ImageFolder(test_dir, transform=transforms)

