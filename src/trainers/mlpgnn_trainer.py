from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import DataParallel
from progress.bar import Bar
from utils.utils import AverageMeter, zero_normalization, get_function_acc, get_Redundant_class_acc

_loss_factory = {
    # Regression
    'l1': nn.L1Loss,
    'sl1': nn.SmoothL1Loss,
    'l2': nn.MSELoss,
    
    # Classification
    'bce': nn.BCELoss,
}

class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, reg_loss, cls_loss, gpus, device):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.reg_loss = reg_loss
        self.cls_loss = cls_loss
        self.gpus = gpus
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.zero = torch.zeros(1).to(self.device)
        self.zero.requires_grad = False
        
    def forward(self, batch):
        preds, max_sim = self.model(batch)
        hs, hf, prob, is_rc, has_redundant_fault = preds
        
        #------------------------------Task 1: Probability Prediction begin------------------------------
        prob_loss = self.reg_loss(prob.to(self.device), batch.prob.to(self.device))
        #------------------------------Task 1: Probability Prediction endin------------------------------
        
        #------------------------------Task 2: Structural Prediction begin-------------------------------
        rc_loss = self.cls_loss(is_rc.to(self.device), batch.is_rc.to(self.device)) 
        #------------------------------Task 2: Structural Prediction endin-------------------------------
        
        #------------------------------Task 3: Function Prediction begin---------------------------------
        node_a = hf[batch.tt_pair_index[0]]
        node_b = hf[batch.tt_pair_index[1]]
        emb_dis = 1 - torch.cosine_similarity(node_a, node_b, eps=1e-8)
        emb_dis_z = zero_normalization(emb_dis)
        tt_dis_z = zero_normalization(batch.tt_dis)
        func_loss = self.reg_loss(emb_dis_z.to(self.device), tt_dis_z.to(self.device))
        #------------------------------Task 3: Function Prediction endin----------------------------------
        
        #------------------------------Task 4: Redundant Fault Prediction begin--------------------------
        # if len(batch.has_redundant_fault) > 0: 
        #     selected_elements = has_redundant_fault[batch.has_redundant_fault.to(torch.long)].to(self.device)
        #     redundant_loss = torch.sum(torch.norm(selected_elements - torch.ones_like(selected_elements)))/ selected_elements.size(0)
        # else:
        #     redundant_loss = self.zero
        # print(has_redundant_fault[0])
        # print(batch.has_redundant_fault[0])
        weights = torch.zeros_like(batch.has_redundant_fault).float().cuda()
        weights = torch.fill_(weights,0.0000)
        weights[batch.has_redundant_fault == [0.0,1.0]] = 1.0
    
        redundant_loss = nn.BCELoss(weight=weights.to(self.device))(has_redundant_fault.to(self.device),batch.has_redundant_fault.to(self.device),)
        # print(redundant_loss)
        #------------------------------Task 4: Redundant Fault Prediction endin---------------------------
        
        loss_stats = {'LProb': prob_loss, 'LRC': rc_loss, 'LFunc': func_loss, "LRFault": redundant_loss}
        
        return hs, hf, has_redundant_fault, loss_stats

class MLPGNNTrainer(object):
    def __init__(self, args, model, optimizer=None):
        self.args = args
        self.optimizer = optimizer
        self.loss_stats, self.reg_loss, self.cls_loss = self._get_losses(args.reg_loss, args.cls_loss)
        self.reg_loss = self.reg_loss.to(self.args.device)
        self.cls_loss = self.cls_loss.to(self.args.device)
        self.model_with_loss = ModelWithLoss(model, self.reg_loss, self.cls_loss, args.gpus, args.device)
    
    def set_weight(self, w_prob, w_rc, w_func, w_rfault):
        self.args.Prob_weight = w_prob
        self.args.RC_weight = w_rc
        self.args.Func_weight = w_func
        self.args.Redundant_weight = w_rfault
        
    def set_device(self, device, local_rank, gpus):
        if len(gpus)> 1:
            self.model_with_loss = self.model_with_loss.to(device)
            self.model_with_loss = nn.parallel.DistributedDataParallel(self.model_with_loss,
                                                                       device_ids=[local_rank], 
                                                                       find_unused_parameters=True)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, dataset, local_rank):
        model_with_loss = self.model_with_loss
        #--------------------------- phase setting
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.args.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()
        #--------------------------- init
        args = self.args
        results = {}
        acc_func_list = []
        precision_redundant_list = []
        recall_redundant_list = []
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(dataset) if args.num_iters < 0 else args.num_iters
        if local_rank == 0:
            bar = Bar('\033[1;31;42m{}/{}'.format(args.task, args.exp_id), max=num_iters)
        end = time.time()
        #--------------------------- iter
        for iter_id, batch in enumerate(dataset):
            if iter_id >= num_iters:
                break
            if len(self.args.gpus) == 1:
                batch = batch.to(self.args.device)
                
            data_time.update(time.time() - end)
            #--------------------------- loss calculate
            hs, hf, has_redundant_fault_pre, loss_stats = model_with_loss(batch)
            loss =  loss_stats['LProb'] * args.Prob_weight + \
                    loss_stats['LRC'] * args.RC_weight + \
                    loss_stats['LFunc'] * args.Func_weight + \
                    loss_stats['LRFault'] * args.Redundant_weight
            loss /= (args.Prob_weight + args.RC_weight + args.Func_weight + args.Redundant_weight)
            loss = loss.mean()
            loss_stats['loss'] = loss 
            #--------------------------- gradient calculate
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model_with_loss.parameters(), args.grad_clip)
                self.optimizer.step()
            #--------------------------- log print
            batch_time.update(time.time() - end)
            end = time.time()
            if local_rank == 0:
                Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                    epoch, iter_id, num_iters, phase=phase,
                    total=bar.elapsed_td, eta=bar.eta_td)
                for l in avg_loss_stats:
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch.num_graphs * len(hf))
                    Bar.suffix = Bar.suffix + \
                        '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
                
                if not args.hide_data_time:
                    Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                        '|Net {bt.avg:.3f}s\033[0m'.format(dt=data_time, bt=batch_time)
                # Get Acc
                if phase == '  val':
                    acc_func = get_function_acc(batch, hf)
                    precision,recall = get_Redundant_class_acc(batch.has_redundant_fault, has_redundant_fault_pre)
                    Bar.suffix = Bar.suffix + '\033[1;31;42m |Acc_func {:}%%\033[0m'.format(acc_func*100)
                    Bar.suffix = Bar.suffix + '\033[1;31;42m |Precision_redundant {:}%%\033[0m'.format(precision*100)
                    Bar.suffix = Bar.suffix + '\033[1;31;42m |Recall_redundant {:}%%\033[0m'.format(recall*100)
                    acc_func_list.append(acc_func)
                    recall_redundant_list.append(recall)
                    precision_redundant_list.append(precision)
                if args.print_iter > 0:
                    if iter_id % args.print_iter == 0:
                        print('{}/{}| {}\033[0m'.format(args.task, args.exp_id, Bar.suffix))
                else:
                    bar.next()
                    
            del hs, hf, loss, loss_stats


        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        if local_rank == 0:
            bar.finish()
            ret['time'] = bar.elapsed_td.total_seconds() / 60.
            if phase == '  val':
                ret['ACC'] = np.average(acc_func)
                ret['Precision_redundant'] =  np.average(precision_redundant_list)
                ret['Recall_redundant'] =  np.average(recall_redundant_list)
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, reg_loss, cls_loss):
        if reg_loss in _loss_factory.keys():
            reg_loss_func = _loss_factory[reg_loss]()
        if cls_loss in _loss_factory.keys():
            cls_loss_func = _loss_factory[cls_loss]()
        loss_states = ['loss', 'LProb', 'LRC', 'LFunc', 'LRFault']
        return loss_states, reg_loss_func, cls_loss_func

    def val(self, epoch, data_loader, local_rank):
        return self.run_epoch('  val', epoch, data_loader, local_rank)

    def train(self, epoch, data_loader, local_rank):
        return self.run_epoch('train', epoch, data_loader, local_rank)
