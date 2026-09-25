from torchvision import datasets
import torchvision.transforms as transforms


def get_dataset(dataset_name, path='/database'):
    if dataset_name in ['amazon', 'dslr', 'webcam']:  # OFFICE-31
    # if dataset_name in ['Art', 'Clipart', 'Product', 'Real_World']:  # OFFICE-Home
    # if dataset_name in ['train', 'validation']:  # VisDA
    # if dataset_name in ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']:  # Domainnet
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        p = path + '/office31/Original_images/' + dataset_name + '/images/'   # Office-31
        # p = path + '/office-home/' + dataset_name     # Office-Home
        # p = path + '/VisDA2017/'                      # VisDA
        # p = path + '/Domainnet/'  # VisDA
        print(p)

        tr_dataset = datasets.ImageFolder(path + '/office31/Original_images/' + dataset_name + '/images/', data_transforms['train'])  # Office-31   # gard_cam
        te_dataset = datasets.ImageFolder(path + '/office31/Original_images/' + dataset_name + '/images/', data_transforms['test'])  # Office-31

        # te_dataset = datasets.ImageFolder('/media/ubuntu/7d17c4ae-0255-4946-a82e-1ebcb5295708/zjt/Dataset/images/', data_transforms['test'])  # Gard_CAM

        # tr_dataset = datasets.ImageFolder(path + '/office-home/' + dataset_name, data_transforms['train']) # Office-Home
        # te_dataset = datasets.ImageFolder(path + '/office-home/' + dataset_name, data_transforms['test'])  # Office-Home

        # tr_dataset = datasets.ImageFolder(path + '/VisDA2017/' + dataset_name, data_transforms['train'])  # VisDA
        # te_dataset = datasets.ImageFolder(path + '/VisDA2017/' + dataset_name, data_transforms['test'])  # VisDA

        # tr_dataset = datasets.ImageFolder(path + '/domainnet/' + dataset_name, data_transforms['train'])  # domainnet
        # te_dataset = datasets.ImageFolder(path + '/domainnet/' + dataset_name, data_transforms['test'])  # domainnet

        # print('{} train set size: {}'.format(dataset_name, len(tr_dataset)))
        print('{} test set size: {}'.format(dataset_name, len(te_dataset)))

    else:
        raise ValueError('Dataset %s not found!' % dataset_name)

    return tr_dataset, te_dataset
