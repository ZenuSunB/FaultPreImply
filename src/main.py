from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import torch
from torch_geometric import data
from torch_geometric.loader import DataLoader, DataListLoader
from config import get_parse_args
from models.model import create_model, load_model, save_model
from trainers.train_factory import train_factory

from utils.logger import Logger
from utils.random_seed import set_seed
from utils.circuit_utils import check_difference
from datasets.mlpgate_dataset import MLPGateDataset

def main(args):
    print('\033[32m====> Using settings {}\033[0m'.format(args))
    #-------------------------------Device begin-------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str
    args.device = torch.device('cuda:0' if args.gpus[0] >= 0 else 'cpu')
    args.world_size = 1
    args.rank = 0  # global rank
    if args.device != 'cpu' and len(args.gpus) > 1:
        args.distributed = len(args.gpus)
    else:
        args.distributed = False
    #-----------------------distributed setting
    if args.distributed:
        if 'LOCAL_RANK' in os.environ:
            args.local_rank = int(os.getenv('LOCAL_RANK'))
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print('\033[32m====> Training in distributed mode. Device {}, Process {:}, total {:}.\033[0m'.format(
            args.device, args.rank, args.world_size))
    else:
        print('\033[32m====> Training in single device: \033[0m', args.device)
    #-----------------------pretrained model dir setting
    if args.local_rank == 0:
        logger = Logger(args, args.local_rank)
        load_model_path = os.path.join(args.save_dir, 'model_last.pth')
        if args.resume and not os.path.exists(load_model_path):
            if args.pretrained_path == '':
                raise "No pretrained model (.pth) found"
            else:
                shutil.copy(args.pretrained_path, load_model_path)
                print('\033[32m====> Copy pth from: \033[0m', args.pretrained_path)
    #-------------------------------Device endin-------------------------------
    
    #-------------------------------datasets begin-------------------------------
    if args.local_rank == 0:
        print('\033[32m====> Loading dataset from: \033[0m', args.data_dir)
    #-----------------------dataset cutting
    all_dataset = MLPGateDataset(args.data_dir, args)
    all_dataset_after_perm = all_dataset[torch.randperm(len(all_dataset))]
    data_len = len(all_dataset)
    if args.local_rank == 0:
        print("All dataset Size: ", data_len)
        print('Splitting the dataset into training and validation sets..')
    training_cutoff = int(data_len * args.trainval_split)
    if args.local_rank == 0:
        print('# training circuits: ', training_cutoff)
        print('# validation circuits: ', data_len - training_cutoff)
    train_dataset = []
    val_dataset = []
    train_dataset = all_dataset_after_perm[:training_cutoff]
    val_dataset = all_dataset_after_perm[training_cutoff:]

    #-----------------------dataLoader setting
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=args.rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=args.rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers, sampler=train_sampler)    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            sampler=val_sampler)
    #-------------------------------datasets endin-------------------------------
    
    #-------------------------------model begin-------------------------------
    model = create_model(args)
    if args.local_rank == 0:
        print('\033[32m====> Creating model...\033[0m')
        print(model)
    #-----------------------optimizer setting 
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    #-----------------------load resume mseg
    start_epoch = 0
    if args.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, args.load_model, optimizer, args.resume, args.lr, args.lr_step, args.local_rank, args.device)
    #-----------------------trainer setting and init
    Trainer = train_factory[args.arch]
    trainer = Trainer(args, model, optimizer)
    trainer.set_device(args.device, args.local_rank, args.gpus)
    #-----------------------val only process
    if args.val_only:
        log_dict_val, _ = trainer.val(0, val_loader)
        return
    #-------------------------------model endin-------------------------------
    
    #-------------------------------train begin-------------------------------
    if args.local_rank == 0:
        print('\033[32m====> Starting training...\033[0m')
    best = 1e10
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        mark = epoch if args.save_all else 'last'
        train_loader.sampler.set_epoch(epoch)
        log_dict_train, _ = trainer.train(epoch, train_loader, args.local_rank)
        #------------------mesg print & log save
        if args.local_rank == 0:
            logger.write('epoch: {} |'.format(epoch), args.local_rank)
            for k, v in log_dict_train.items():
                logger.scalar_summary('train_{}'.format(k), v, epoch, args.local_rank)
                logger.write('{} {:8f} | '.format(k, v), args.local_rank)
            if args.save_intervals > 0 and epoch % args.save_intervals == 0:
                save_model(os.path.join(args.save_dir, 'model_{}.pth'.format(mark)),
                           epoch, model, optimizer)
        #------------------run val
        with torch.no_grad():
            val_loader.sampler.set_epoch(0)
            log_dict_val, _ = trainer.val(epoch, val_loader, args.local_rank)
        #------------------model save & log save
        if args.local_rank == 0:
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch, args.local_rank)
                logger.write('{} {:8f} | '.format(k, v), args.local_rank)
            if log_dict_val[args.metric] < best:
                best = log_dict_val[args.metric]
                save_model(os.path.join(args.save_dir, 'model_best.pth'),
                           epoch, model)
            else:
                save_model(os.path.join(args.save_dir, 'model_last.pth'),
                           epoch, model, optimizer)
            logger.write('\n', args.local_rank)
        #------------------lr update & model save 
        if epoch in args.lr_step:
            if args.local_rank == 0:
                save_model(os.path.join(args.save_dir, 'model_{}.pth'.format(epoch)),
                           epoch, model, optimizer)
            lr = args.lr * (0.1 ** (args.lr_step.index(epoch) + 1))
            if args.local_rank == 0:
                print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    #-------------------------------train endin-------------------------------
    if args.local_rank == 0:
        logger.close()
            
if __name__ == '__main__':
    args = get_parse_args()
    set_seed(args)
    main(args)
