'''attach libs'''
import torch
import numpy as np
import scipy.stats as ss

class byz_attack:
    def attacked_grad(self):
        raise NotImplementedError
    def __str__(self):
        raise NotImplementedError
    def default_grad_copy_attack(self, model):
        for p in model.parameters():
            if p.requires_grad:
                
                if int(p.grad_batch.size()[0]) == self.honest_worker_num:
                    honest_grad = p.grad_batch
                    if self.honest_worker_num >= self.mal_worker_num:
                        mal_grad = p.grad_batch[-self.mal_worker_num:]
                    else:
                        mal_grad = honest_grad[torch.randint(0, self.honest_worker_num, (self.mal_worker_num,))]
                    p.grad_batch = torch.cat([honest_grad, mal_grad], dim=0)
                else: 
                    pass
                    honest_grad = p.grad_batch[:self.honest_worker_num]
                    if self.honest_worker_num >= self.mal_worker_num:
                        mal_grad = p.grad_batch[-self.mal_worker_num:]
                    else:
                        mal_grad = honest_grad[torch.randint(0, self.honest_worker_num, (self.mal_worker_num,))]
                    p.grad_batch = torch.cat([honest_grad, mal_grad], dim=0)

                assert self.honest_worker_num + self.mal_worker_num == int(p.grad_batch.size()[0])


class no_attack(byz_attack):
    def __init__(self, mal_worker_num = None, honest_worker_num = None):
        self.mal_worker_num = mal_worker_num
        self.honest_worker_num = honest_worker_num
        self.attacker_name = 'nobyz'

    def attacked_grad(self, model, **kwargs):
        return

    def __str__(self):
        return f'there is no attack'

class label_flipping_attack(byz_attack):
    def __init__(self, mal_worker_num = None, honest_worker_num = None):
        #print('/n==> under label flipping attack/n')
        self.mal_worker_num = mal_worker_num
        self.honest_worker_num = honest_worker_num
        self.attacker_name = 'lf'

    def attacked_grad(self, model, **kwargs):
        if kwargs['iter_num'] < kwargs['start_attack']:
            self.default_grad_copy_attack(model)
            

    def __str__(self):
        return f'label flipping attack with malicious worker number:{self.mal_worker_num}'

class gaussian_attack(byz_attack):
    def __init__(self, mean = 0, std = 1, mal_worker_num = None, honest_worker_num = None):
        self.mean = mean
        self.std = std
        self.mal_worker_num = mal_worker_num
        self.honest_worker_num = honest_worker_num

        self.attacker_name = 'gaussian'

    def attacked_grad(self, model, **kwargs):
        if self.mal_worker_num == 0:
            return
        if kwargs['iter_num'] < kwargs['start_attack']:
            self.default_grad_copy_attack(model)
            return
        '''treate the last mal_worker_num worker as malicious worker'''
        for p in model.parameters():
            if p.requires_grad:

                if int(p.grad_batch.size()[0]) == self.honest_worker_num:
                    honest_grad = p.grad_batch
                else:
                    honest_grad = p.grad_batch[:self.honest_worker_num]

                mal_grad = torch.randn(self.mal_worker_num, *(list(p.grad_batch.size())[1:]), device = p.grad_batch.device) * self.std + self.mean

                if int(p.grad_batch.size()[0]) == self.honest_worker_num:
                    p.grad_batch = torch.cat([honest_grad, mal_grad], dim=0)
                else:
                    p.grad_batch[-self.mal_worker_num:] = mal_grad

                assert int(list(p.grad_batch.size())[0]) == self.honest_worker_num + self.mal_worker_num, (int(list(p.grad_batch.size())[0]), self.honest_worker_num + self.mal_worker_num)

    def __str__(self):
        return f'gaussian_attack with mean:{self.mean}, std:{self.std} and malicious worker number:{self.mal_worker_num}'

class local_attack(byz_attack):
    def __init__(self, mal_worker_num = None, honest_worker_num = None):
        self.mal_worker_num = mal_worker_num
        self.honest_worker_num = honest_worker_num
        self.attacker_name = 'local'

    def attacked_grad(self, model, **kwargs):
        if self.mal_worker_num == 0:
            return
        if kwargs['iter_num'] < kwargs['start_attack']:
            self.default_grad_copy_attack(model)
            return

        gama = self.mal_worker_num / self.honest_worker_num**0.5 - 1
        assert gama > 0, gama
        '''treate the last mal_worker_num worker as malicious worker'''
        for p in model.parameters():
            if p.requires_grad:

                if int(p.grad_batch.size()[0]) == self.honest_worker_num:
                    honest_grad = p.grad_batch
                else:
                    honest_grad = p.grad_batch[:self.honest_worker_num]

                result = -(gama+1) * torch.sum(honest_grad, dim=0) / self.mal_worker_num
                repeat_dims = [self.mal_worker_num] + [ 1 for i in list(p.grad_batch.size())[1:] ]
                mal_grad = result.repeat(*repeat_dims) 

                if int(p.grad_batch.size()[0]) == self.honest_worker_num:
                    p.grad_batch = torch.cat([honest_grad, mal_grad], dim=0)
                else:
                    p.grad_batch[-self.mal_worker_num:] = mal_grad

                assert int(list(p.grad_batch.size())[0]) == self.honest_worker_num + self.mal_worker_num, (int(list(p.grad_batch.size())[0]), self.honest_worker_num + self.mal_worker_num)

    def __str__(self):
        return f'local attack with malicious worker number:{self.mal_worker_num}'
