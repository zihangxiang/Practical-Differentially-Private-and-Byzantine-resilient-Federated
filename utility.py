import torch
import enum
import math
import os
from pathlib import Path
import csv
import time
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
#
from backpack import backpack, extend
from backpack.extensions import BatchGrad
#
import privacy_analysis.handler as ph
import logger


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help = "the dataset", type=str, default=" ", required=True)
    parser.add_argument("--epsilon", help = "the epsilon value", type=str, default=" ", required=True)
    parser.add_argument("--DP_mode", help = "local or central DP setting", type=str, default=" ", required=True)
    parser.add_argument("--seed", help = "seed number", type=int, default=" ", required=True)
    parser.add_argument("--mal_worker_portion", help = "mal worker portion", type=str, default=" ", required=True)
    parser.add_argument("--anti_byz", help = "perform anti byzantine", type=str, default=" ", required=True)
    parser.add_argument("--non_iid", help = "activate non_iid setting", type=str, default=" ", required=True)
    parser.add_argument("--att_key", help = "attacker description", type=str, default=" ", required=True)
    parser.add_argument("--start_att", help = "the time point of the attack start to attack", type=str, default=" ", required=True)
    parser.add_argument("--base_lr", help = "base learing rate", type=str, default=" ", required=True)
    return parser.parse_args()

class Phase(enum.Enum):
    TRAINING = TRAIN = enum.auto()
    VALIDATION = VALID = VAL = enum.auto()
    TESTING = TEST = enum.auto()

phase_name_dict = {
                    Phase.TRAINING: "Training",
                    Phase.VALIDATION: "Validation",
                    Phase.TESTING: "Testing",
                    }

class ClassificationMetrics:
    """Accumulate per-class confusion matrices for a classification task."""
    metrics = ('accur', 'recall', 'specif', 'precis', 'npv', 'f1_s', 'iou')

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.tp = self.fn = self.fp = self.tn = 0
        self.hit_count = 0
        self.hit_accuracy = 0
        self.num_of_prediction = 0

    @property
    def count(self):  # a.k.a. actual positive class
        """Get the number of samples per-class."""
        return self.tp + self.fn

    @property
    def frequency(self):
        """Get the per-class frequency."""
        # we avoid dividing by zero using: max(denominator, 1)
        count = self.tp + self.fn
        return count / count.sum().clamp(min=1)

    @property
    def total(self):
        """Get the total number of samples."""
        # return self.count.sum()
        return ( self.tp + self.fn ).sum()

    @torch.no_grad()
    def update(self, pred, true):
        """Update the confusion matrix with the given predictions."""
        pred, true = pred.flatten(), true.flatten()
        classes = torch.arange(0, self.num_classes, device=true.device)
        valid = (0 <= true) & (true < self.num_classes)
        '''
        this trick:
        pred_pos is n * 1 tensor, pred is 1 * n tensor
        '''
        pred_pos = classes.view(-1, 1) == pred[valid].view(1, -1)
        positive = classes.view(-1, 1) == true[valid].view(1, -1)
        pred_neg, negative = ~pred_pos, ~positive
        self.tp += (pred_pos & positive).sum(dim=1)
        self.fp += (pred_pos & negative).sum(dim=1)
        self.fn += (pred_neg & positive).sum(dim=1)
        self.tn += (pred_neg & negative).sum(dim=1)
        
        self.hit_count += (pred == true).sum().item()
        self.num_of_prediction += int(pred.numel())
        
        self.hit_accuracy = self.hit_count / self.num_of_prediction
    def reset(self):
        """Reset all accumulated metrics."""
        self.tp = self.fn = self.fp = self.tn = 0

    @property
    def accur(self):
        """Get the per-class accuracy."""
        # we avoid dividing by zero using: max(denominator, 1)
        return (self.tp + self.tn) / self.total.clamp(min=1)
    

    @property
    def recall(self):
        """Get the per-class recall."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tp / (self.tp + self.fn).clamp(min=1)
    
    @property
    def specif(self):
        """Get the per-class recall."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tn / (self.tn + self.fp).clamp(min=1)
    
    @property
    def npv(self):
        """Get the per-class recall."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tn / (self.tn + self.fn).clamp(min=1)

    @property
    def precis(self):
        """Get the per-class precision."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tp / (self.tp + self.fp).clamp(min=1)

    @property
    def f1_s(self):  # a.k.a. Sorensen–Dice Coefficient
        """Get the per-class F1 score."""
        # we avoid dividing by zero using: max(denominator, 1)
        tp2 = 2 * self.tp
        return tp2 / (tp2 + self.fp + self.fn).clamp(min=1)

    @property
    def iou(self):
        """Get the per-class intersection over union."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tp / (self.tp + self.fp + self.fn).clamp(min=1)

    def weighted(self, scores):
        """Compute the weighted sum of per-class metrics."""
        return (self.frequency * scores).sum()

    def __getattr__(self, name):
         """Quick hack to add mean and weighted properties."""
         if name.startswith('mean_') or not name.startswith(
             'mean_') and name.startswith('weighted_'):
              metric = getattr(self, '_'.join(name.split('_')[1:]))
              return metric.mean() if name.startswith('mean_') else self.weighted(metric)
         raise AttributeError(name)

    def __repr__(self):
        """A tabular representation of the metrics."""
        metrics = torch.stack([getattr(self, m) for m in self.metrics])
        perc = lambda x: f'{float(x) * 100:.2f}%'.ljust(8)
        out = 'Class'.ljust(6) + ''.join(map(lambda x: x.ljust(8), self.metrics))

        if self.num_classes > 20:
            return self._total_summary(metrics, perc)

        out += '\n' + '-' * 60
        for i, values in enumerate(metrics.t()):
            out += '\n' + str(i).ljust(6)
            out += ''.join(map(lambda x: perc(x.mean()), values))

        return out + self._total_summary(metrics, perc)

    def _total_summary(self, metrics, perc):
        out = ''
        out += '\n' + '-' * 60

        out += '\n'+'Mean'.ljust(6)
        out += ''.join(map(lambda x: perc(x.mean()), metrics))

        out += '\n'+'Wted'.ljust(6)
        out += ''.join(map(lambda x: perc(self.weighted(x)), metrics))
        out += '\n' + 'hit accuracy: ' + f'{float(self.hit_accuracy) * 100:.2f}%'
        return out

    def disp(self, with_detail = True):
        if with_detail:
            print( self )
        else:
            metrics = torch.stack([getattr(self, m) for m in self.metrics])
            perc = lambda x: f'{float(x) * 100:.2f}%'.ljust(8)
            print(self._total_summary(metrics, perc))

setup = {}
class log_master:
    def __init__(self, root='logs'):
       
        self.root = f'{os.getcwd()}/{root}'
        if root not in os.listdir(os.getcwd()):
            os.makedirs(self.root, exist_ok = False)

        self.filename_existed = set()

    def csv_writing(self, filename, content):

        with open(f'{self.root}/{filename}', 'a') as file:
            writer = csv.writer(file)
            if filename not in self.filename_existed:
                self.filename_existed.add(filename)

                writer.writerow([])
                writer.writerow([
                                    str( time.strftime('[%d_%H_%M_%S]',time.localtime(time.time())) ) + 
                                    ' ==> new recording'
                                    ])
                writer.writerow( ['setup:'])
                _ = [ writer.writerow( [f'       {i} -> { str(setup[i]) }'] ) for i in setup ]

            writer.writerow(content)

    
    def csv_reading(self, filename):
         try:
              content = []
              with open(f'{self.root}/{filename}', 'r') as file:
                   reader = csv.reader(file)
                   content.extend(iter(reader))
              return content
         except Exception as ex:
              print(f"csv_reading failes, due to:{str(ex)}")


def element_wise_std(std):
    ''' maybe it will be implemented '''
    pass


def load_checkpoint(filename, model, optimizer = None):
    checkpoint_path =  Path( os.getcwd() ) / 'best_model'
    if filename not in os.listdir(checkpoint_path):
        print('no pt model can be found')
        return False

    checkpoint_path = checkpoint_path / filename
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return True

def save_checkpoint(model_state, filename, only_save_model = False):
    checkpoint_path = Path( os.getcwd() ) / 'best_model'
    if only_save_model:
        torch.save(model_state['model'], checkpoint_path / filename )
    else:
        torch.save(model_state, checkpoint_path / filename )

def accuracy(output, targets):
    predictions = output.argmax(dim=1, keepdim=True).view_as(targets)
    return predictions.eq(targets).float().mean().item()

lr_idx = 0
def one_epoch(epoch, phase, model, loader, device, optimizer = None, loss_metric = None):
    metrics = ClassificationMetrics(num_classes = model.num_of_classes)
    metrics.num_images = metrics.loss = 0  # adds extra attributes
    training = phase is Phase.TRAINING

    with torch.set_grad_enabled(training):
        model.train(training)

        s = time.time()
        for batch in tqdm(loader):
            inputs, targets = map(lambda x: x.to(device), batch)

            '''flip the lables of malicious workers
               only when it is in the training routine, for validation and testing, no flip is conducted'''
            if setup['attacker'].attacker_name == 'lf' and training and BRP_simulator.start_attack_ite <= BRP_simulator.iteration_step + 1:
                attacker_idx = len(targets) - setup['attacker'].mal_worker_num
                ori_targets_mal = torch.clone(targets[attacker_idx:])
                targets[attacker_idx:] = model.num_of_classes - 1 - targets[attacker_idx:]
                tmp = list(targets[len(targets)-setup['attacker'].mal_worker_num:] )
                targets[len(targets)-setup['attacker'].mal_worker_num:] = torch.tensor(tmp)

            logits = model(inputs)
            loss = loss_metric( logits, targets.flatten() )

            if training:
                optimizer.zero_grad()

                if setup['is_DP_private'] or setup['is_anti_byzantine_aggregation']:
                        with backpack(BatchGrad()):
                            loss.backward()
                        BRP_simulator.simulate()
                else:
                        loss.backward()  
                
                optimizer.step()
                global the_scheduler
                if the_scheduler is not None:
                    the_scheduler.step()

            '''recover the true lables
               only when it is in the training routine, for validation and testing, no flip is conducted'''
            if setup['attacker'].attacker_name == 'lf' and training and BRP_simulator.start_attack_ite <= BRP_simulator.iteration_step:
                targets[len(targets)-setup['attacker'].mal_worker_num:] = torch.clone(ori_targets_mal)

            '''update batch metrics'''
            metrics.num_images += len(inputs)
            metrics.loss += loss.item() * len(inputs)
            metrics.update(logits.data.argmax(dim=1), targets.flatten())

    logger.write_log(f'>>>>>            time spent for one epoch: {round(time.time()-s,2)} s            <<<<<')
    metrics.loss /= metrics.num_images
    return metrics

def train_or_validate(epoch, loader, phase, model, 
                     device, optimizer, loss_metric):
     if loader is None:
          print('empty loader...')
          return 

     metrics = one_epoch(epoch, phase, model, loader, device, optimizer, loss_metric = loss_metric)
     record_data_type = 'weighted_recall'
     logger.write_log( f'{phase_name_dict[phase]}: weighted_recall = { round(float(metrics.__getattr__(record_data_type))*100, 2) }%' )
     return metrics

def _see_model_total_para(model, verbose = True):
     total = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
     if verbose: logger.write_log(f'==> total number of parameters are: {total}')
     return total


def _log_the_model_performance(the_logger, epoch, train_metrics, val_metrics, test_metrics, record_data_type = 'weighted_recall'):
     the_logger.csv_writing('loss.csv', [
                                         epoch, 
                                         "%.3f"%float(train_metrics.loss) if train_metrics else 'NAN',
                                         "%.3f"%float(val_metrics.loss) if val_metrics else 'NAN',
                                         "%.3f"%float(test_metrics.loss) if test_metrics else 'NAN', 
                                         ])
     the_logger.csv_writing(f'{record_data_type}.csv', [
                                                        epoch,
                                                        f'{ float( train_metrics.__getattr__(record_data_type) ) * 100:.2f}%'
                                                        if train_metrics else 'NAN',
                                                        f'{ float( val_metrics.__getattr__(record_data_type) ) * 100:.2f}%'
                                                        if val_metrics else 'NAN',
                                                        f'{ float( test_metrics.__getattr__(record_data_type) ) * 100:.2f}%'
                                                        if test_metrics else 'NAN',
                                                    ])
BRP_simulator = None
the_scheduler = None
def train(model, loss_metric, loaders, optimizer, scheduler, device, total_epoch, **kwargs):
    the_logger = log_master(root = 'logs')

    global the_scheduler
    the_scheduler = scheduler

    global setup
    setup = {}

    setup['model size'] = _see_model_total_para(model)


    for key in kwargs:
        setup[key] = kwargs[key]

    for key in setup:
        logger.write_log(f'{key}: {setup[key]}', c_tag='[setup]')
    logger.write_log(f'[{setup["dataset"]}]-[{setup["mal_worker_portion"]}]-[{str(setup["attacker"])[:4]}]-[{setup["epsilon"]}]-[{setup["seed"]}]', c_tag='[query]')

    '''loaders = (train_loader, val_loader, test_loader)'''
    '''these variable are recorded epoch-wisely'''
    best_state = None
    logs = []
    loss_metric = loss_metric

    '''loaders'''
    train_loader, _, test_loader = loaders[0], loaders[1], loaders[2]

    '''extend model to use backpack libs'''
    model = extend(model)
    if setup['is_DP_private'] == False and setup['is_anti_byzantine_aggregation'] == False:
        pass
    else:
        if setup['is_central_DP_BR']:
            logger.write_log('==> is_central_DP_BR <==', c_tag ='[mode]')
        elif setup['is_local_DP_BR']:
            logger.write_log('==> is_local_DP_BR <==', c_tag ='[mode]')
        train_loader = ph.to_special_dataloader(
                                                loader = train_loader, 
                                                sampling_type = setup['sampling_type'],
                                                sampling_rate = setup['sampling_rate'],
                                                local_batch_size = setup['local_batch_size'],
                                                attacker = setup['attacker'],
                                                honest_worker_num = setup['attacker'].honest_worker_num, 
                                                mal_worker_num = setup['attacker'].mal_worker_num, 
                                                num_of_classes = model.num_of_classes,
                                                pub_data_size = setup['pub_data_size'],
                                                pub_data_batch_size = setup['pub_data_batch_size'],
                                                non_iid = setup['non_iid'],
                                                fix_worker_data_distributing_non_iid = setup['fix_worker_data_distributing_non_iid'],
                                               
                                                ) 

    setup_seed(setup['seed'])

    '''center_clipped_byzantine_and_privacy'''
    global BRP_simulator
    assert setup['is_central_DP_BR'] != setup['is_local_DP_BR']

    BRP_simulator = setup['simulator'](
                                        model = model,
                                        worker_momentum_beta = setup['the_worker_momentum'],
                                        worker_num = setup['num_workers'],
                                        attacker = setup['attacker'],
                                        start_attack_ite = setup['start_attack_ite'],
                                        is_anti_byzantine_aggregation = setup['is_anti_byzantine_aggregation'],
                                        is_DP_private = setup['is_DP_private'],
                                        is_central_DP = setup['is_central_DP_BR'],
                                        std = setup['std'],
                                        pub_data_batch_size = setup['pub_data_batch_size'],
                                        local_batch_size = setup['local_batch_size'],
                                        total_epoch = total_epoch,
                                        lr = setup['lr'],
                                        loss_metric = loss_metric,
                                        device = device,
                                        sampling_rate = setup['sampling_rate'],
                                        optimizer = optimizer,
                                        )

    for epoch in range(total_epoch):
        # if epoch > 0:
        #     break
        logger.write_log(f'\n\nEpoch: {epoch}'.ljust(11) + '#' * 70)

        '''training'''
        train_metrics = train_or_validate(epoch, train_loader, Phase.TRAINING, 
                            model, device, optimizer, loss_metric = loss_metric) 


        '''test, just for looking at testset, determining when to stop the training will not depend on the testset'''
        test_metrics = train_or_validate(epoch, test_loader, Phase.TESTING, 
                            model, device, optimizer, loss_metric = loss_metric)
        

        '''logging the training result of current epoch '''
        record_data_type = 'weighted_' + 'recall'
        _log_the_model_performance(the_logger, epoch, 
                                    train_metrics, None, test_metrics, record_data_type)

        '''early stopping if loss is NaN (not a number) or infinity'''
        if math.isnan(train_metrics.loss) or math.isinf(train_metrics.loss):
            logger.write_log('Reached invalid loss! (no point in continuing)')
            logger.write_log('train_metrics.loss',train_metrics.loss)
            break
    
    logger.write_log(f'==> [parameter norm]: {BRP_simulator.max_parameter_norm}')
    logger.write_log(f'==> [min_noise_dominance_factor]: {BRP_simulator.min_noise_dominance_factor}')
    
    return best_state, logs
