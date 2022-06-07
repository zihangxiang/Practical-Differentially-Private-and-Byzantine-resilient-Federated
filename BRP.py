import torch
import logger
from scipy import stats

class FL_simulator:
    def __init__(self,  model = None, 
                        worker_momentum_beta = None, 
                        worker_num = None, 
                        attacker = None, 
                        start_attack_ite = None,
                        is_anti_byzantine_aggregation = False,
                        is_DP_private = False,
                        is_central_DP = None,
                        std = None,
                        pub_data_batch_size = None,
                        local_batch_size = None,
                        total_epoch = None,
                        lr = None,
                        loss_metric = None,
                        device = None,
                        sampling_rate = None,
                        optimizer = None,
                        ):

        self.boosting_rounds = 0
        self.is_central_DP = is_central_DP
        self.sampling_rate = sampling_rate 
        self.optimizer = optimizer
        self.optimizer_modified_up = False      
        self.optimizer_modified_down = False  

        self.lr_scale = 100
        self.server_lr = self.optimizer.param_groups[0]["lr"] / self.lr_scale
        

        self.loss_metric = loss_metric
        self.device = device

        self.model = model
        self.reloaded = False

        tensor_dic = {}
        for submodule in self.model.modules():
            for s in submodule.parameters():
                if s.requires_grad:
                    if id(s) not in tensor_dic:
                        tensor_dic[id(s)] = 0
                    if isinstance(submodule, torch.nn.Linear):
                            tensor_dic[id(s)] = 1

        self.tensor_dic = tensor_dic

        self.total_epoch = total_epoch
        self.iteration_step = 0
        self.lr = lr

        total = 0
        cnn_total = 0
        linear_total = 0

        self.which_param_req_grad = set()
        self.param_holder = []
        for p in model.parameters():
            if p.requires_grad:
                self.which_param_req_grad.add(id(p))
                self.param_holder.append(None)
                total += int(p.numel())
                if tensor_dic[id(p)] == 0:
                    cnn_total += int(p.numel())
                if tensor_dic[id(p)] == 1:
                    linear_total += int(p.numel())

        self.cnn_total = cnn_total
        logger.write_log(f'==> CNN parameter numbers: {self.cnn_total}' )
        self.linear_total = linear_total
        logger.write_log(f'==> linear parameter numbers: {self.linear_total}' )
        self.total_params = total
        logger.write_log(f'==> total parameter numbers: {self.total_params}' )

        self.worker_momentum_beta = worker_momentum_beta
        self.server_momentum_beta = 0.9
        self.worker_num = worker_num

        self.attacker = attacker
        self.start_attack_ite = start_attack_ite
        self.is_anti_byzantine_aggregation = is_anti_byzantine_aggregation

        self.is_DP_private = is_DP_private

        self.std = std

        self.pub_data_batch_size = pub_data_batch_size
        
        self.local_batch_size = local_batch_size

        self.last_time_client_grad = [0 for p in self.model.parameters() if p.requires_grad]
        self.last_time_server_grad = [0 for p in self.model.parameters() if p.requires_grad]


        self.last_time_score = 0

        self.this_time_server_grad = [ None for p in self.model.parameters() if p.requires_grad]
        self.this_time_server_grad_no_momentum = [ None for p in self.model.parameters() if p.requires_grad]
        self.worker_grad_accu = [ 0 for p in self.model.parameters() if p.requires_grad]
        self.server_grad_accu = [ 0 for p in self.model.parameters() if p.requires_grad]
        self.last_time_server_anchor = [ 0 for p in self.model.parameters() if p.requires_grad]

        self.local_noise_with_shared_seed = [ 0 for p in self.model.parameters() if p.requires_grad]
        self.local_noise_with_shared_seed_for_mal = [ 0 for p in self.model.parameters() if p.requires_grad]
        self.shared_noise_std = 0

        self.mal_grad_holder = [ 0 for p in self.model.parameters() if p.requires_grad]

        # self.at_least_choose_portion = 0.5

        self.honest_portion = self.attacker.honest_worker_num / self.worker_num

        std = (
                self.std / (self.worker_num *  self.honest_portion)**0.5\
                if self.is_central_DP else self.std
                ) / self.local_batch_size

        self.prior_norm, self.min_prior_norm, self.max_prior_norm = self.chi_square_norm(std)   
        logger.write_log(f'==> prior norm: {self.prior_norm}')
        logger.write_log(f'==> min_prior_norm: {self.min_prior_norm}')
        logger.write_log(f'==> max_prior_norm: {self.max_prior_norm}')

        self.max_parameter_norm = -1
        self.min_noise_dominance_factor = 1000000


        self.size_info = [p.data.size() for p in self.model.parameters() if p.requires_grad]

        self.final_std = (self.std / (self.worker_num * self.honest_portion)**0.5 
                            if self.is_central_DP else self.std) / self.local_batch_size 

        self.honest_total = self.attacker.honest_worker_num * self.local_batch_size

        self.benign_score_list = []
        self.mal_score_list = []

        self.b_select_portion = []
        self.m_select_portion = []

    def take_a_look_at_grad_norm(self, info):
        print(info, self.compute_per_grad_norm() + 1e-6)

    def recover_per_example_grad(self):
        
        parameter_norm = sum([p.data.norm()**2 for p in self.model.parameters() if p.requires_grad])**0.5
        if parameter_norm > self.max_parameter_norm:
            self.max_parameter_norm = parameter_norm

        for p in self.model.parameters():
            if p.requires_grad:
                p.grad_batch = p.grad_batch * int( p.grad_batch.size()[0] )

    def checking_the_clipping_operation(self, houlder_tester_to_ensure_norm_is_clipped, C):
        '''ensuring that the per example norm is clipped'''
        '''holding the subparts of the model parameters' norm squared: tester_to_ensure_norm_is_clipped'''

        l2_clipped_norm_square_per = torch.stack(houlder_tester_to_ensure_norm_is_clipped)
        total_clipped_norms = (torch.sum(l2_clipped_norm_square_per, dim=0))**0.5
        test_result = total_clipped_norms <= C*1.001
        if not torch.all(test_result).item():
            #logger.write_log(torch.all(test_result), test_result)
            logger.write_log('\n--> the outlier norm is:')
            logger.write_log(total_clipped_norms[test_result==False])
            raise ValueError('norm clipping failed!')

    def _make_broadcastable(self, tensor_to_be_reshape, target_tensor):
        broadcasting_shape = (-1, *[1 for _ in target_tensor.shape[1:]])
        return tensor_to_be_reshape.reshape(broadcasting_shape)

    def separate_server_grad(self):
        '''server action'''
        for index, p in enumerate(self.model.parameters()):
            if p.requires_grad:
                self.this_time_server_grad[index] = torch.sum(p.grad_batch[ : self.pub_data_batch_size ], dim = 0
                                                             ) / self.pub_data_batch_size 
                # ''' server momentum '''
                server_beta = 0.99
                self.this_time_server_grad[index] = (1 - server_beta) * self.this_time_server_grad[index] + server_beta * self.last_time_server_grad[index]
                self.last_time_server_grad[index] = torch.clone(self.this_time_server_grad[index])

        ''' normalizing the server grad '''
        server_grad_norm = sum((sub_tensor.norm()) ** 2 for sub_tensor in self.this_time_server_grad)** 0.5 + 1e-6
        for index in range(len(self.this_time_server_grad)):
            self.this_time_server_grad[index] = self.this_time_server_grad[index] / server_grad_norm

        ''' make grad_batch only hold local grad '''
        for p in self.model.parameters():
            if p.requires_grad:
                p.grad_batch = p.grad_batch[ self.pub_data_batch_size : ]

    def __do_worker_momentum(self):
        ''' compute local grad with momentum ''' 
        
        if self.worker_momentum_beta > 0:   
            for index, p in enumerate(self.model.parameters()):
                if p.requires_grad:
                    p.grad_batch[:self.honest_total] =  (1 - self.worker_momentum_beta) * p.grad_batch[:self.honest_total] + self.worker_momentum_beta * self.last_time_client_grad[index]

        ''' save mal worker grad '''
        if not self.is_anti_byzantine_aggregation:
            for index, p in enumerate(self.model.parameters()):
                if p.requires_grad:
                    self.mal_grad_holder[index] = p.grad_batch[self.honest_total:]

    def simulating_worker_grad_computation(self):
        self.iteration_step += 1

        self.__do_worker_momentum()

        self.per_grad_normalizing()

        '''get the batch size'''
        '''it is the same as data example number, which can be greater than num of workers'''
        for p in self.model.parameters():
            if p.requires_grad:
                b_size = int( p.grad_batch.size()[0] )
                break

        if b_size == self.attacker.honest_worker_num * self.local_batch_size + self.attacker.mal_worker_num and self.attacker.mal_worker_num > 0:
            self.only_honest_worker_grad = False
        elif b_size == self.attacker.honest_worker_num * self.local_batch_size:
            self.only_honest_worker_grad = True
        else:
            raise ValueError(f'b_size is not correct, b_size{b_size}, honest_worker_num{self.attacker.honest_worker_num}, mal_worker_num{self.attacker.mal_worker_num}, local_batch_size{self.local_batch_size}')
        
        if self.only_honest_worker_grad:
            for p in self.model.parameters():
                if p.requires_grad:
                    ''' now, p.grad_batch should only contain grad from local agent, grad from server are removed'''
                    all_chunks = torch.chunk(p.grad_batch, self.attacker.honest_worker_num, dim = 0)
                    p.grad_batch = torch.stack( [ torch.sum(t, dim = 0) for t in all_chunks ] )
                    assert int( p.grad_batch.size()[0] ) == self.attacker.honest_worker_num
        else:
             for p in self.model.parameters():
                if p.requires_grad:
                    ''' now, p.grad_batch should only contain grad from local agent, grad from server are removed'''
                    all_chunks = torch.chunk(p.grad_batch[:self.honest_total], self.attacker.honest_worker_num, dim = 0)
                    p.grad_batch = torch.cat(
                                                [
                                                torch.stack( [ torch.sum(t, dim = 0) for t in all_chunks ] ),
                                                p.grad_batch[-self.attacker.mal_worker_num:] 
                                                ], 
                                                dim=0
                                            )
                    assert int( p.grad_batch.size()[0] ) == self.worker_num, f'{p.grad_batch.size()[0]} != {self.worker_num}'


    def per_grad_normalizing(self, norm = 1):
        ''' normalizing the grad '''
        local_grad_norm = (self.compute_per_grad_norm() + 1e-6) / norm
        
        tester_to_ensure_norm_is_clipped = []
        for p in self.model.parameters():
            if p.requires_grad:
                p.grad_batch = p.grad_batch / self._make_broadcastable(local_grad_norm, p.grad_batch)# per datapoint clipping
                '''ensuring the norm is normalized'''
 

    def check_all_client_norm_within_prior(self):
        all_norm = self.compute_per_grad_norm()
        tmp_flag = torch.zeros_like(all_norm)

        tmp_flag[all_norm<self.min_prior_norm] = 1
        tmp_flag[all_norm>self.max_prior_norm] = 1

        if torch.sum(tmp_flag) > 0:
            logger.write_log(f'[warning:] some norm is not within the range, {all_norm[tmp_flag==1]}')

    def check_ks_test(self):
        tmp = 0
        for p in self.model.parameters():
            if p.requires_grad:
                if isinstance(tmp, int):
                    tmp = p.grad_batch.flatten(start_dim=1)
                else:
                    tmp = torch.cat((tmp, p.grad_batch.flatten(start_dim=1)), dim=1)
    
        std = (self.std / (self.worker_num * self.honest_portion)**0.5 
                            if self.is_central_DP else self.std) / self.local_batch_size 

        _, pvalue_0 = stats.ks_2samp( tmp[0].cpu().numpy(), (torch.randn_like(tmp[0])*std).cpu().numpy() )
        _, pvalue_m1 = stats.ks_2samp( tmp[-1].cpu().numpy(), (torch.randn_like(tmp[0])*std).cpu().numpy() )

        logger.write_log( f'[ks_test, pvalue[0]]: {pvalue_0}' )
        logger.write_log( f'[ks_test, pvalue[-1]]: {pvalue_m1}' )

    def attacker_makes_attack(self):
        if self.attacker.mal_worker_num == 0:
            return

        '''before sending the grads to central server, make mal attacks'''
        self.attacker.attacked_grad(
                                    self.model, 
                                    iter_num = self.iteration_step, 
                                    start_attack = self.start_attack_ite,
                                    std = self.final_std,
                                    local_batch_size = self.local_batch_size,
                                    total_params = self.total_params,
                                    )


        if not self.is_anti_byzantine_aggregation:
            for index, p in enumerate(self.model.parameters()):
                if p.requires_grad:
                    if self.attacker.attacker_name == 'sign':
                        p.grad_batch[self.attacker.honest_worker_num:] = -self.mal_grad_holder[index]
                    else:
                        p.grad_batch[self.attacker.honest_worker_num:] = self.mal_grad_holder[index]
        '''-----> from now on the grad is sent to the server ------>''' 

    def chi_square_norm(self, std):
        if std == 0:
            return 0
        square_mean, the_std= (self.total_params * std**2), (2 * self.total_params * std**4)**0.5
        mean = (square_mean + 3 * the_std)**0.5
        return mean, (square_mean - 8 * the_std)**0.5, (square_mean + 8 * the_std)**0.5
            
    
    def compute_per_grad_norm(self, ):
        l2_norms_all_params_list = []
        for p in self.model.parameters():
            if p.requires_grad:
                local_grad = p.grad_batch
                dims = tuple(range( 1,len( local_grad.size() ) ))
                l2_norms_all_params_list.append( ( torch.linalg.norm(local_grad, dim = dims) )**2 )
        l2_norms_all_params = torch.stack(l2_norms_all_params_list)
        return torch.sqrt( torch.sum(l2_norms_all_params, dim = 0) )

    def make_aggregation(self):
        ''''''
        if self.is_anti_byzantine_aggregation:

            score = self.compute_local_score()
            score_sum = torch.sum(score)

            for index, p in enumerate(self.model.parameters()):
                    if p.requires_grad:
                        p.grad = torch.sum(p.grad_batch[score > 0], dim = 0) / score_sum
                                                                                         
        else: 
            ''' DP is done locally, no need to do DP here, if no anti byzantine, just do average '''
            for index, p in enumerate(self.model.parameters()):
                if p.requires_grad:
                    p.grad = torch.sum(p.grad_batch[:self.attacker.honest_worker_num], dim = 0) / self.attacker.honest_worker_num

                    if self.is_central_DP:
                        self.last_time_client_grad[index] = torch.clone(p.grad)
                    else: # aleady done
                        pass

    def __str__(self):
        return 'central_DP_simulator' if self.is_central_DP else 'local_DP_simulator'

    def let_workers_compute_grad(self):
        self.recover_per_example_grad()
        self.separate_server_grad()
        self.simulating_worker_grad_computation()
        self.__local_grad_noise_injection()

    def check_real_coordinate_wise_NDF(self):
        tmp = 0
        for p in self.model.parameters():
                if p.requires_grad:
                    if isinstance(tmp, int):
                        tmp = p.grad_batch.flatten(start_dim=1)
                    else:
                        tmp = torch.cat((tmp, p.grad_batch.flatten(start_dim=1)), dim=1)

        logger.write_log(f"flattened grad size: {tmp.size()}")
        coordinate_wise_value = abs(tmp).max()

        c_noise_dominance_factor = self.std / (self.worker_num * self.honest_portion)**0.5 / coordinate_wise_value \
                                if self.is_central_DP \
                                else self.std / coordinate_wise_value

        mean_c_noise_dominance_factor = torch.mean( self.std / (self.worker_num * self.honest_portion)**0.5 / abs(tmp) \
                                if self.is_central_DP \
                                else self.std / abs(tmp), dim=1)

        logger.write_log(f"\n\n[real min coordinate wise NDF]: {c_noise_dominance_factor}")
        logger.write_log(f"[mean coordinate wise NDF]: {mean_c_noise_dominance_factor}")
        logger.write_log(f"[emprical min coordinate wise NDF]: {self.min_noise_dominance_factor}")
        logger.write_log(f"[grad coorinate mean and std]: {torch.mean(tmp, dim=1)}, {torch.std(tmp, dim=1)}")
        logger.write_log(f"[noise std]: {self.std / (self.worker_num * self.honest_portion)**0.5 if self.is_central_DP else self.std}\n\n")
                                            
    def __local_grad_noise_injection(self):

        ''' min noise dominance factor across clients '''
        max_local_norm = torch.max( self.compute_per_grad_norm() + 1e-6 )
        coordinate_wise_value = ( max_local_norm**2 / self.total_params )**0.5 
        noise_dominance_factor = ( self.std / (self.worker_num * self.honest_portion)**0.5
                                   if self.is_central_DP
                                   else self.std ) / coordinate_wise_value

        if noise_dominance_factor < self.min_noise_dominance_factor:
            self.min_noise_dominance_factor = noise_dominance_factor

        # self.check_real_coordinate_wise_NDF()

        if self.is_central_DP:
            for p in self.model.parameters():
                if p.requires_grad:
                    noise = torch.randn_like(p.grad_batch) * self.std / (self.worker_num * self.honest_portion)**0.5 #* self.local_batch_size
                    p.grad_batch = (p.grad_batch + noise ) / self.local_batch_size
        else:
            for index, p in enumerate(self.model.parameters()):
                if p.requires_grad:
                    noise = torch.randn_like(p.grad_batch) * self.std #* self.local_batch_size
                    p.grad_batch = (p.grad_batch + noise) / self.local_batch_size #* (1-self.worker_momentum_beta)

                    if self.worker_momentum_beta > 0:
                        if not self.only_honest_worker_grad:
                            self.last_time_client_grad[index] = torch.repeat_interleave( p.grad_batch[:self.attacker.honest_worker_num], self.local_batch_size, dim = 0)

    def simulate(self):
        self.let_workers_compute_grad()
        self.attacker_makes_attack()
        # self.grad_sent_to_server()
        self.make_aggregation() 

    def compute_local_score(self):

        summation_subparts = []
        for index, p in enumerate(self.model.parameters()):
            if p.requires_grad:
                subparts_inner_prod = p.grad_batch * self.this_time_server_grad[index]
                remain_dims = tuple(range( 1,len( p.grad_batch.size() ) ))
                ''' row vector, each entry is a partial sum houlder for inner product to one local worker'''
                tmp = torch.sum(subparts_inner_prod, dim = remain_dims)
                summation_subparts.append(tmp)
        score_change_per_local_subparts = torch.stack(summation_subparts)
        score = torch.sum( score_change_per_local_subparts, dim = 0 ) / self.compute_per_grad_norm() * 10

        ''' score acc '''
        large, large_index = torch.topk( score, int( len(score) * self.honest_portion ), largest = True)

        loc = torch.bitwise_and( score < large.mean(), score > -100 )
        score[loc] = 0

        score_acc = score + self.last_time_score 
        self.last_time_score = torch.clone(score_acc)

        ''' estimation part '''
        large, large_index = torch.topk( self.last_time_score, int( len(score) * self.honest_portion ), largest = True)
        the_mean = large.mean()
        e_std = large.std()

        score_submit = torch.zeros_like(score_acc)
        score_submit[large_index] = 1

        if self.iteration_step % int(1/self.sampling_rate) == 0: 
            # logger.write_log(f'==> lr: {self.optimizer.param_groups[0]["lr"]}')
            B_part = self.last_time_score[:self.attacker.honest_worker_num]
            logger.write_log(f'\n[B score]: {B_part}')
            logger.write_log(f'\n[B score]: min:{round(float(torch.min(B_part)), 3)}, max:{round(float(torch.max(B_part)),3)}, avg:{round(float(torch.mean(B_part)),3)}, std:{round(float(torch.std(B_part)),3)}' )
            logger.write_log(f'\n[B score]: {score_submit[:self.attacker.honest_worker_num]}')
            logger.write_log(f'\n[B score]: {int(sum(score_submit[:self.attacker.honest_worker_num]))}/{self.attacker.honest_worker_num}')

            M_part = self.last_time_score[-self.attacker.mal_worker_num:]
            logger.write_log(f'\n\n[M score]: {M_part}')
            logger.write_log(f'\n\n[M score]: min:{round(float(torch.min(M_part)),3)}, max:{round(float(torch.max(M_part)),3)}, avg:{round(float(torch.mean(M_part)),3)}, std:{round(float(torch.std(M_part)),3)}' )
            logger.write_log(f'\n[M score]: {score_submit[-self.attacker.mal_worker_num:]}')
            logger.write_log(f'\n[M score]:{int(sum(score_submit[-self.attacker.mal_worker_num:]))}/{self.attacker.mal_worker_num}')
            
            logger.write_log(f'\n==> [mean: {the_mean}, esti_std: {e_std}]\n')
            logger.write_log(f'==> [parameter norm]: {self.max_parameter_norm}')
            logger.write_log(f'==> [min_noise_dominance_factor]: {self.min_noise_dominance_factor}\n')
        
        return score_submit


            
