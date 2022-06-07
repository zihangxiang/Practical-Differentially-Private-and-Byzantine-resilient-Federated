import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader, Dataset
from pathlib import Path
import os
from PIL import Image
#
from . import dataset_setup
###############################################################################################
print('-----------> Using colorectal_histology data')
data_file_root =  Path( dataset_setup.get_dataset_data_path() ) / f'DATASET_DATA/Kather_texture_2016_image_tiles_5000'
print('==> dataset located at: ', data_file_root)
num_of_classes = 8


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_metric = torch.nn.functional.cross_entropy
loss_metric = torch.nn.CrossEntropyLoss()


class MyDataset(Dataset):                            
    def __init__(self, data_file_root, train, transform):
        self.train = train
        self.transform = transform
        self.data_file_root = data_file_root

        class_folder_names = [ '01_TUMOR', '02_STROMA', '03_COMPLEX', '04_LYMPHO', '05_DEBRIS', '06_MUCOSA', '07_ADIPOSE', '08_EMPTY']
        assert num_of_classes == len(class_folder_names)
        self.class_label = class_folder_names

        each_class_data_root = [self.data_file_root / foldername for foldername in class_folder_names]

        file_name_holder_for_each_class = [] 
        for i in range(num_of_classes):
            file_name_holder_for_each_class.append(
                                                 [
                                                 filename 
                                                 for filename 
                                                 in os.listdir(each_class_data_root[i])
                                                 if filename.endswith('.tif')
                                                 ]
                                                )

        self.total_len = sum([len(each_name_holder) for each_name_holder in file_name_holder_for_each_class])
        
        self.total_name_holder = []
        for index, folder_name in enumerate(class_folder_names):
            for file_name in file_name_holder_for_each_class[index]:
                self.total_name_holder.append( {
                                                'file_name': folder_name + '/' + file_name, 
                                                'label': index
                                                } )
        assert len(self.total_name_holder) == self.total_len 


    def __getitem__(self, index):
       
        file_name = self.data_file_root / self.total_name_holder[index]['file_name']
        label = self.total_name_holder[index]['label']
        # print(file_name)
        return  self.transform(Image.open(file_name)), label

    def __len__(self):
        return self.total_len



###############################################################################################
#
transformation = T.Compose([
                            T.Resize( size = (64,64) ),
                            T.ToTensor(),
                            T.Normalize(
                                        (0.6499, 0.4723, 0.5844), (0.1511, 0.1592, 0.1381)
                                         ),

                          ])

## dataset setup

def get_all_dataset(seed = None):
    dataset = MyDataset(data_file_root, train = True, transform = transformation)
    ForTest = 400

    if seed is not None:
        dataset_train, dataset_val = random_split(dataset, [len(dataset) - ForTest, ForTest],
                                                    generator=torch.Generator().manual_seed(seed)) #[59000, 1000] (dataset, [30000, 30000]) 
        dataset_val, dataset_test =  random_split(dataset_val, [len(dataset_val) - ForTest+1, ForTest-1],
                                                    generator=torch.Generator().manual_seed(seed)) 
    else:
        dataset_train, dataset_val = random_split(dataset, [len(dataset) - ForTest, ForTest],) #[59000, 1000] (dataset, [30000, 30000]) 
        dataset_val, dataset_test =  random_split(dataset_val, [len(dataset_val) - ForTest+1, ForTest-1],)
                                    

          
    return dataset_train, dataset_val, dataset_test


def get_all(batchsize_train = None, seed = None):
    dataset_train, dataset_val, dataset_test = get_all_dataset(seed = seed)

    #training loader
    dataloader_train = DataLoader(
                                dataset = dataset_train,
                                batch_size = batchsize_train,
                                shuffle = True,
                                num_workers = 4,
                                pin_memory = (device.type == 'cuda'),
                                drop_last = True,
                                )

    #validation loader
    dataloader_val = DataLoader(
                                dataset = dataset_val,
                                batch_size = 256,
                                shuffle = True,
                                num_workers = 4,
                                pin_memory = (device.type == 'cuda'),
                                drop_last = False,
                                )
    #testing loader
    dataloader_test = DataLoader(
                                dataset = dataset_test,
                                batch_size = 256,
                                shuffle = True,
                                num_workers = 4,
                                pin_memory = (device.type == 'cuda'),
                                drop_last = False,
                                )

    return (dataset_train, dataset_val, dataset_test), (dataloader_train, dataloader_val, dataloader_test)
    

'''model setup'''
#################################################################################################
class model(nn.Module):
    def __init__(self, num_of_classes):
        super().__init__()  
        self.num_of_classes = num_of_classes

        # self.block_1 = resnet8(num_of_classes)

        ''' '''
        non_linear = nn.ELU
        n_groups = 4
        def res_bloack(inter_channel):
            
            def unit_block(inter_channel):
                return [
                        nn.Conv2d(inter_channel, inter_channel, kernel_size = 3, stride = 1, padding = 1),
                        non_linear(),
                        nn.GroupNorm(n_groups, inter_channel, affine = False),
                       ]
            return nn.Sequential( 
                                 *unit_block(inter_channel),
                                 *unit_block(inter_channel),

                                )

        inter_channel = 16

        self.block_0 = nn.Sequential( 
                                      nn.Conv2d(3, inter_channel, kernel_size = 3, stride = 1, padding = 1),
                                      non_linear(),
                                      nn.GroupNorm(n_groups, inter_channel, affine = False),
                                       )
        self.inter_block_list = nn.Sequential( *[res_bloack(inter_channel) for _ in range(7)] )                          
        self.f_block = nn.Sequential(
                                    nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Flatten(),

                                    nn.Linear(inter_channel * 1 , 32),
                                    non_linear(),
                                    nn.Linear(32, self.num_of_classes),
                                    )


        dataset_setup.init_model_para(self.block_0)
        for block in self.inter_block_list:
            dataset_setup.init_model_para(block)
        dataset_setup.init_model_para(self.f_block)


    def forward(self, x):
        
        x = self.block_0(x)

        for i in range(len(self.inter_block_list)):
            x_tmp = x
            x = self.inter_block_list[i](x)
            x = nn.MaxPool2d(kernel_size = 2, stride = 1)(x + x_tmp)

        x = self.f_block(x)
        return x

        # return self.block_1(x)

##################################################################################################
