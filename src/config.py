import os
import argparse

def get_parse_args():
    parser = argparse.ArgumentParser(description='Pytorch training script of DeepGate.')
    #--------------------------------------------basic settings begin--------------------------------------------
    parser.add_argument('task', default='prob', choices=['prob', 'diff', 'redundant'],
                        help='prob | diff | redundant ')
    parser.add_argument('--exp_id', default='default')
    parser.add_argument('--load_model', default='',
                        help='path to pretrained model')
    parser.add_argument('--pretrained_path', default='../exp/pretrained/pretrain_l1/model_last.pth', type=str)
    parser.add_argument('--resume', action='store_true',
                        help='resume an experiment. '
                            'Reloaded the optimizer parameter and '
                            'set load_model to model_last.pth '
                            'in the exp dir if load_model is empty.')
    #--------------------------------------------basic settings endin--------------------------------------------
    
    #--------------------------------------------system setting begin--------------------------------------------
    parser.add_argument('--gpus', default='-1', 
                        help='-1 for CPU, use comma for multiple gpus')
    parser.add_argument('--random_seed', type=int, default=208, 
                        help='random seed')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='dataloader threads. 0 for single-thread.')
    parser.add_argument('--not_cuda_benchmark', action='store_true',
                        help='disable when the input size is not fixed.')
    #--------------------------------------------system setting endin--------------------------------------------
    
    #--------------------------------------------dataset settings begin------------------------------------------
    parser.add_argument('--no_rc', default=False, action='store_true')
    parser.add_argument('--reload_dataset', default=False, action='store_true', help='Reload inmemory data')
    parser.add_argument('--small_train', default=False, 
                        action='store_true',help='if True, use a smaller version of train set')
    parser.add_argument('--data_dir', default='../data/random_circuits',
                        type=str, help='the path to the dataset')
    parser.add_argument('--reconv_skip_connection', default=False, 
                        action='store_true', help='construct the skip connection between source node and the reconvergence node.')
    parser.add_argument('--no_node_cop', default=False, 
                        action='store_true', help='not to use the C1 values as the node features')
    parser.add_argument('--un_directed', default=False, action='store_true', 
                        help='If true, model the circuit as the undirected graph. Default: circuit as DAG')
    parser.add_argument('--node_reconv', default=False, 
                        action='store_true', help='use the reconvergence info as the node features')
    parser.add_argument('--logic_implication', default=False, 
                        action='store_true', help='use the logic implication/masking as an additonal node feature or not.')
    parser.add_argument('--enable_aig', default=True, action='store_true')      # default enable aig, no support MIG now 
    #--------------------------------------------dataset settings endin----------------------------------

    #--------------------------------------------model setting begin-------------------------------------
    parser.add_argument('--arch', default='mlpgnn', choices=['recgnn', 'convgnn', 'dagconvgnn', 'mlpgnn', 'mlpgnn_merge'],
                        help='model architecture. Currently support'
                        'recgnn | convgnn ' 
                        'recgnn will updata the embedding in T(time) dim, while convgnn will update the embedding in K(layer) dim.'
                        'recgnn corresponds to dagnn/dvae settings, which considers DAG circuits.')
    parser.add_argument('--activation_layer', default='relu', type=str, choices=['relu', 'relu6', 'sigmoid'],
                        help='The activation function to use in the FC layers.')  
    parser.add_argument('--norm_layer', default='batchnorm', type=str,
                        help='The normalization function to use in the FC layers.')
    parser.add_argument('--num_fc', default=3, type=int,
                        help='The number of FC layers')     
    
    parser.add_argument('--num_rounds', type=int, default=1, metavar='N',
                        help='The number of rounds for grn propagation.'
                        '1 - the setting used in DAGNN/D-VAE')
    parser.add_argument('--predict_diff', default=False, 
                        action='store_true', help='predict the difference between the simulated ground-truth probability and C1.')
    parser.add_argument('--intermediate_supervision', action='store_true', default=False,
                        help='Calculate the losses for every round.')
    parser.add_argument('--no_reverse', action='store_true', default=False,
                        help='Not to use the reverse layer to propagate the message.')
    parser.add_argument('--custom_backward', action='store_true', default=False,
                        help='Whether to use the custom backward or not.')
    parser.add_argument('--mask', action='store_true', default=False,
                        help='Use the mask for the node embedding or not')
    
    parser.add_argument('--num_aggr', default=3, type=int,
                        help='the number of aggregation layers.')
    parser.add_argument('--dim_hidden', type=int, default=64, metavar='N',
                        help='hidden size of recurrent unit.')
    parser.add_argument('--dim_mlp', type=int, default=32, metavar='N',
                        help='hidden size of readout layers') 
    parser.add_argument('--dim_pred', type=int, default=1, metavar='N',
                        help='hidden size of readout layers')
    parser.add_argument('--wx_update', action='store_true', default=False,
                        help='The inputs for the update function considers the node feature of mlp.')
    parser.add_argument('--wx_mlp', action='store_true', default=False,
                        help='The inputs for the mlp considers the node feature of mlp.')        
    parser.add_argument('--dim_edge_feature', default=16,
                        type=int, help='the dimension of node features')
    parser.add_argument('--aggr_function', default='tfmlp', type=str, choices=['deepset', 'aggnconv', 'gated_sum', 'conv_sum', 'mlp', 'attnmlp', 'tfmlp'],
                        help='the aggregation function to use.')
    parser.add_argument('--disable_encode', action='store_true', default=False)
    #--------------------------------------------model setting endin-------------------------------------
    
    #--------------------------------------------train and val begin-------------------------------------
    parser.add_argument('--lr', type=float, default=1.0e-4, 
                        help='learning rate for batch size 32.')
    parser.add_argument('--lr_step', type=str, default='30,45',
                        help='drop learning rate by 10.')
    parser.add_argument('--weight_decay', type=float, default=1e-10, 
                        help='weight decay (default: 1e-10)')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='total training epochs.')
    parser.add_argument('--num_iters', type=int, default=-1,
                        help='default: #samples / batch_size.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--grad_clip', type=float, default=0.,
                        help='gradiant clipping')
    parser.add_argument('--trainval_split', default=0.80, type=float,
                        help='the splitting setting for training dataset and validation dataset.')
    parser.add_argument('--val_only', action='store_true', 
                        help='Do the validation evaluation only.')
    #--------------------------------------------train and val endin-------------------------------------
    
    #-----------------------------------------loss settings begin------------------------------------
    parser.add_argument('--prob_loss', action='store_true', default=False,
                            help='To use the simulated probabilities as complementary supervision.')
    parser.add_argument('--reg_loss', default='l1',
                             help='regression loss: sl1 | l1 | l2 | focalloss')
    parser.add_argument('--cls_loss', default='bce',
                             help='classification loss: bce - BCELoss | bce_logit - BCELossWithLogit | cross - CrossEntropyLoss')
    #-----------------------------------------loss settings endin------------------------------------
    
    #-----------------------------------------loss weight settings begin------------------------------------
    parser.add_argument('--Prob_weight', type=float, default=5)
    parser.add_argument('--RC_weight', type=float, default=3)
    parser.add_argument('--Func_weight', type=float, default=1)      
    parser.add_argument('--Redundant_weight', type=float, default=1)    
    #-----------------------------------------loss weight settings endin------------------------------------
    
    #-----------------------------------------log settings begin------------------------------------
    parser.add_argument('--print_iter', type=int, default=0, 
                            help='disable progress bar and print to screen.')
    parser.add_argument('--hide_data_time', action='store_true',
                             help='not display time during training.')
    parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
    parser.add_argument('--save_intervals', type=int, default=5,
                             help='number of epochs to run validation.')
    parser.add_argument('--metric', default='loss', 
                             help='main metric to save best model')
    #-----------------------------------------log settings endin------------------------------------
    
    args = parser.parse_args()
    args.gpus_str = args.gpus
    args.gpus = [int(gpu) for gpu in args.gpus.split(',')]
    args.gpus = [i for i in range(len(args.gpus))] if args.gpus[0] >=0 else [-1]
    args.lr_step = [int(i) for i in args.lr_step.split(',')]
    
    args.circuit_file = "graphs.npz"
    args.label_file = "labels.npz"
    
    args.use_edge_attr = False
    
    if args.enable_aig:
        args.gate_to_index = {'INPUT': 0, 'AND': 1, 'NOT': 2}
    else:
        args.gate_to_index = {'INPUT': 0, 'GND': 1, 'VDD': 2, 'MAJ': 3, 'NOT': 4, 'BUF': 5}
        
    args.num_gate_types = len(args.gate_to_index)
    args.dim_node_feature = len(args.gate_to_index)
    args.reverse = not args.no_reverse
    
    #-----------------------------------------dir setting begin-----------------------------------------
    args.root_dir = os.path.join(os.path.dirname(__file__), '..')
    args.exp_dir = os.path.join(args.root_dir, 'exp', args.task)
    args.save_dir = os.path.join(args.exp_dir, args.exp_id)
    args.debug_dir = os.path.join(args.save_dir, 'debug')
    if args.resume and args.load_model == '':
        model_path = args.save_dir
        args.load_model = os.path.join(model_path, 'model_last.pth')
    elif args.load_model != '':
        model_path = args.save_dir
        args.load_model = os.path.join(model_path, args.load_model)
    #-----------------------------------------dir setting endin-----------------------------------------

    args.local_rank = 0
    return args