import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import RandomSampler
import numpy as np
import random
#
import privacy_analysis as pa
   

def to_special_dataloader( *, 
                        loader,
                        sampling_type ,
                        sampling_rate,
                        local_batch_size,
                        attacker,
                        honest_worker_num,
                        mal_worker_num ,
                        num_of_classes ,
                        pub_data_size,
                        pub_data_batch_size,
                        non_iid ,
                        fix_worker_data_distributing_non_iid,
                        ): 

        
    return DataLoader(
                        dataset = MyDataset(
                                            base_dataset = loader.dataset, 
                                            sampling_type = sampling_type, 
                                            sampling_rate = sampling_rate,
                                            num_of_classes = num_of_classes,
                                            pub_data_size = pub_data_size,
                                            pub_data_batch_size = pub_data_batch_size,
                                            local_batch_size = local_batch_size,
                                            mal_worker_num = mal_worker_num,
                                            honest_worker_num = honest_worker_num,
                                            non_iid = non_iid,
                                            fix_worker_data_distributing_non_iid = fix_worker_data_distributing_non_iid,
                                            attacker = attacker
                                            ),
                        batch_size = 1,
                        shuffle = False,
                        num_workers = 4,
                        pin_memory = loader.pin_memory,
                        collate_fn = lambda batch: (batch[0][0], batch[0][1]),
                        )

class MyDataset(Dataset):
    def __init__(self, *, 
                    base_dataset, 
                    sampling_type, 
                    sampling_rate,
                    num_of_classes,
                    pub_data_size,
                    pub_data_batch_size,
                    local_batch_size,
                    mal_worker_num,
                    honest_worker_num,
                    non_iid,
                    fix_worker_data_distributing_non_iid,
                    attacker,
                    ):
        self.sampling_type = sampling_type
        self.base_dataset = base_dataset
        self.batch_indexes_holder = []
        self.data_len = len(self.base_dataset)
        self.iteration_for_each_epoch = int(1/sampling_rate)
        self.num_of_classes = num_of_classes

        self.pub_data_size = pub_data_size
        self.local_batch_size = local_batch_size
        self.mal_worker_num = mal_worker_num
        self.honest_worker_num = honest_worker_num
        self.non_iid = non_iid
        self.fix_worker_data_distributing_non_iid = fix_worker_data_distributing_non_iid
        self.attacker = attacker

        if sampling_type == 'fixed_size':
            raise NotImplementedError('fixed_size sampling is not implemented yet')

        elif sampling_type == 'have_pub_root':
            assert pub_data_size % num_of_classes == 0

            '''simulating sampling for pub data size'''
            print('==> simulating drawing samples for public root data')
            num_of_sample_per_class = pub_data_size // num_of_classes
            assert num_of_sample_per_class > 0
            pub_data_index = []
            pub_data_counter = [0 for i in range(num_of_classes)]
            
            label_list = []
            pub_data_label_to_index_dict = {label:[] for label in range(num_of_classes)}

            permuted_index_list_for_pub_data = np.random.permutation(len(self.base_dataset))
            
            print('==> permuted_index_list_for_pub_data fisrt 10:', permuted_index_list_for_pub_data[:10])
            for data_index in permuted_index_list_for_pub_data: #range( len(self.base_dataset )):
            # for data_index in pub_index_list:
                data_index = int(data_index)
                the_label = int( self.base_dataset[data_index][1] )
                
                if pub_data_counter[the_label] < num_of_sample_per_class:
                    pub_data_counter[the_label] += 1
                    pub_data_index.append(data_index)
                    pub_data_label_to_index_dict[the_label].append(data_index)
                    if the_label not in label_list:
                        label_list.append(the_label)
                    
                if sum(pub_data_counter) == pub_data_size:
                    print('==> Done for simulating drawing samples for public root data')
                    break

            assert len(pub_data_index) == pub_data_size
        
            print('==> data is only distributed in all workers')
            '''if mal_worker_num is set, data is only distributed in honest workers, then '''            

            if non_iid == True:
                print('==> simulating non-iid data distribution...')
            else:
                print('==> simulating iid data distribution...')

            ''' for the label_dict, each key is the label, and the value is the list of index '''
            
            label_dic = {i:[] for i in range(num_of_classes)}
            for data_index in range( len(self.base_dataset )):
                the_label = int( self.base_dataset[data_index][1] )
                label_dic[the_label].append(data_index)

            ''' each worker has some data(index)'''
            data_index_houlder = [[] for i in range(honest_worker_num)]
            label_distribution = [[] for i in range(honest_worker_num)]
            
            ''' form a label dic for each worker, each key is the label, and the value is the list of index '''
            label_to_index_dic_for_each_worker = [ {label: [] for label in range(num_of_classes)} for _ in range(honest_worker_num)]

            if fix_worker_data_distributing_non_iid:
                ''' only for non iid case '''
                torch.manual_seed(1234)

            for label in range(num_of_classes):
                ''' for each label, distribute it to all workers '''
                length_of_data_for_current_label = len(label_dic[label])
                if length_of_data_for_current_label < honest_worker_num:
                    raise ValueError(f'label {label} has number {length_of_data_for_current_label}, which is less than honest_worker_num {honest_worker_num}')

                while True:
                    if non_iid == True:
                        tmp = torch.rand((honest_worker_num,))
                    else:
                        tmp = torch.ones((honest_worker_num,))
                    dist_vector_on_worker = (tmp / sum(tmp)) * length_of_data_for_current_label
                    test = dist_vector_on_worker[dist_vector_on_worker < 1]
                    if len(test) == 0:
                        break

                ''' for each worker, distribute the data which has the same label '''
                for worker_number in range(honest_worker_num):
                    ''' for current label, each worker will get some data '''
                    if worker_number == honest_worker_num - 1:
                        data_index_houlder[worker_number] += label_dic[label]
                        label_to_index_dic_for_each_worker[worker_number][label] += data_index_houlder[worker_number]
                        label_distribution[worker_number].append(len(label_dic[label]))
                    else:
                        num_for_current_worker = round(float(dist_vector_on_worker[worker_number]))
                        data_index_houlder[worker_number] += label_dic[label][:num_for_current_worker]
                        label_to_index_dic_for_each_worker[worker_number][label] = data_index_houlder[worker_number]
                        label_dic[label] = label_dic[label][num_for_current_worker:]
                        label_distribution[worker_number].append(num_for_current_worker)

            if non_iid == True:
                print('non-iid simulation results:')
                for i,item in enumerate(label_distribution):
                    print('worker:', i, item, sum(item))

            # random.seed(2022)

            instantiated_index_list = []
            for item in data_index_houlder:
                instantiated_index_list += item

            each_worker_has_data_num = int(self.data_len / honest_worker_num)
            for i in range(self.iteration_for_each_epoch):
                ''' this is the indexes that should be fetched out in every iteration '''
                index_list_one_round = []

                ''' pub data also have its batch size '''
                ''' first prepend the pubdata to the index holder'''
                ''' sampling without replacement '''
                np.random.shuffle(pub_data_index)
                index_list_one_round += pub_data_index[:pub_data_batch_size]
                
                ''' fixed-size sampling with replacement '''
                ''' no local data partition '''
                for j in range(honest_worker_num):
                    for k in range(local_batch_size):
                        index_list_one_round.append(
                                                    instantiated_index_list[
                                                    random.randint(0, each_worker_has_data_num-1) + 
                                                    j * each_worker_has_data_num
                                                                            ]
                                                    )

                self.batch_indexes_holder.append(index_list_one_round)
                ''' for mal worker '''
                if attacker.attacker_name not in  ['no', 'local']:
                    if mal_worker_num == 0:
                        for _ in range(40):
                            _ = random.randint(0, self.data_len-1)
                    else:
                        for _ in range(mal_worker_num):
                            # for _ in range(local_batch_size):
                            index_list_one_round.append( random.randint(0, self.data_len-1) )

                self.batch_indexes_holder.append(index_list_one_round)
                     
        else:
            raise NotImplementedError(f'this sampling method: {sampling_type}, is not implemented')

    def __getitem__(self, index):

        data_piece = list( [ self.base_dataset[i][0] for i in self.batch_indexes_holder[index] ] )
        data = torch.stack( data_piece )

        labels_piece = list( [ self.base_dataset[i][1] for i in self.batch_indexes_holder[index] ] )
        labels = torch.tensor( labels_piece )
        
        return data, labels
        
    def __len__(self):
        return self.iteration_for_each_epoch
