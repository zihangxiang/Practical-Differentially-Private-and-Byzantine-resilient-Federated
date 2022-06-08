import torch
import math

'''helper libs'''
import utility
import privacy_analysis.accounting_analysis as pa
import byzantine_attacks.attacks as BAs
import BRP

arg_setup = utility.parse_args()

''' setups '''
epsilon = float(arg_setup.epsilon)
mal_worker_portion = float(arg_setup.mal_worker_portion)/100
is_local_DP_BR = (arg_setup.DP_mode == 'localDP')
seed = int(arg_setup.seed)
non_iid = False if int(arg_setup.non_iid) == 0 else True
if not is_local_DP_BR:
    assert arg_setup.DP_mode == 'centralDP'
the_dataset = arg_setup.dataset
att_key = arg_setup.att_key
start_att = float(arg_setup.start_att)
is_anti_byzantine_aggregation = False if int(arg_setup.anti_byz) == 0 else True
is_DP_private = True
base_lr = float(arg_setup.base_lr)

''' seeds '''
utility.setup_seed(seed)

''' dataset/model setup '''
if the_dataset == 'mnist':
    import datasets.mnist as dms
    num_workers = 20
    EPOCH = 8

elif the_dataset == 'fashion':
    import datasets.fashion as dms
    num_workers = 20
    EPOCH = 8

elif the_dataset == 'colorectal':
    import datasets.colorectal as dms
    num_workers = 10
    EPOCH = 10

elif the_dataset == 'usps':
    import datasets.usps as dms
    num_workers = 10
    EPOCH = 10

else:
    raise ValueError('Dataset not specified...')

pub_data_size = dms.num_of_classes * 2
pub_data_batch_size = pub_data_size
local_data_batch_size = 16
the_momentum = 0.1
batchsize_train = num_workers * local_data_batch_size

model = dms.model(num_of_classes = dms.num_of_classes).to(dms.device)
total_params = utility._see_model_total_para(model, verbose = False)

all_dataset, all_loader = dms.get_all(batchsize_train = batchsize_train, seed = 2022)
dataset_train, dataset_val, dataset_test = all_dataset
fix_worker_data_distributing_non_iid = True 

''' noise calculation '''
delta = (1/(len(dataset_train) / num_workers))**1.1 if is_local_DP_BR else (1/len(dataset_train))**1.1
q = 1 / ( len(dataset_train) / batchsize_train ) 
print('\n\n==>sampling rate:', q, "epsilon:", epsilon, "epoch:", EPOCH, 'delta:', delta, )

std = pa.get_std(q = q , EPOCH = EPOCH , epsilon = epsilon, delta = delta) # * (1-the_momentum) 
radius_norm = (total_params * std**2)**0.5

base_epsilon = 2
base_std = pa.get_std(q = q , EPOCH = EPOCH , epsilon = base_epsilon, delta = delta, verbose = False) #* (1-the_momentum) 
base_radius_norm = (total_params * base_std**2)**0.5

lr = base_lr * base_radius_norm / radius_norm
####################################################################################################

honest_worker_num = num_workers
honest_worker_portion = 1 - mal_worker_portion
mal_worker_num = int(honest_worker_num / (1 - mal_worker_portion) * mal_worker_portion)#int( num_workers * mal_worker_portion )
num_workers = honest_worker_num + mal_worker_num

local_batch_size = batchsize_train // honest_worker_num
gaussian_std = std / local_data_batch_size if is_local_DP_BR else std / local_data_batch_size / (num_workers*honest_worker_portion)**0.5
attackers = {
            'nobyz': BAs.no_attack(mal_worker_num = mal_worker_num, honest_worker_num = honest_worker_num),
            'gaussian':BAs.gaussian_attack(mean = 0, std = gaussian_std, mal_worker_num = mal_worker_num, honest_worker_num = honest_worker_num),
            'lf':BAs.label_flipping_attack(mal_worker_num = mal_worker_num, honest_worker_num = honest_worker_num),
            'local':BAs.local_attack(mal_worker_num = mal_worker_num, honest_worker_num = honest_worker_num),
            }

the_attack = attackers[att_key]
print('==> attacker: ', the_attack)

#--------------------------------------#--------------------------------------#
#                               optimizer setup
optimizer = torch.optim.SGD( model.parameters(), lr = lr) 

#scheduer setup
scheduler = None

#--------------------------------------#--------------------------------------#
#                               training routine
best_state, logs = utility.train(
                                model, 
                                dms.loss_metric, 
                                all_loader, 
                                optimizer, scheduler, dms.device, EPOCH,

                                # kwargs below
                                EPOCH = EPOCH,
                                lr = lr,
                                base_lr = base_lr,

                                # DP SPEC 
                                epsilon = epsilon,
                                delta = delta,
                                std = std,

                                # training mode
                                is_DP_private = is_DP_private, 
                                is_anti_byzantine_aggregation = is_anti_byzantine_aggregation,

                                is_central_DP_BR = not is_local_DP_BR,
                                is_local_DP_BR = is_local_DP_BR,

                                sampling_type = 'have_pub_root', 
                                fix_worker_data_distributing_non_iid = fix_worker_data_distributing_non_iid,

                                attacker = the_attack,
                                mal_worker_portion = mal_worker_portion,
                                start_attack_ite = int(start_att * EPOCH / q),
                                start_attack_percent = start_att,

                                # sampler sepc
                                sampling_rate = q, 
                                num_workers = num_workers, 
                                local_batch_size = local_batch_size,

                                pub_data_size = pub_data_size, 
                                pub_data_batch_size = pub_data_batch_size,
                                non_iid = non_iid,

                                # For info recording
                                train_data_len = len(dataset_train),
                                train_batch_size = batchsize_train,

                                simulator = BRP.FL_simulator,
                                the_worker_momentum = the_momentum,

                                dataset = the_dataset,
                                seed = seed,
                                )

