'''
Parse the AIG (in bench format) and truth table for each nodes
16-11-2022
Note: 
    gate_to_index = {'PI': 0, 'AND': 1, 'NOT': 2}
    x_data: 0 - Name, 1 - gate type, 2 - level, 3 - is RC, 4 - RC source node 
'''

import argparse
import glob
import os
import sys
import platform
import time
import numpy as np
from collections import Counter

import utils.circuit_utils as circuit_utils
import utils.utils as utils



data_folds = [ \
            ['/home/zenu/code/FaultPreImply/data/data_raw/data_iscas89/bench','/home/zenu/code/FaultPreImply/data/data_raw/data_iscas89/fault_mesg']\
            ,['/home/zenu/code/FaultPreImply/data/data_raw/data_iscas85/bench','/home/zenu/code/FaultPreImply/data/data_raw/data_iscas85/fault_mesg']\
            ,['/home/zenu/code/FaultPreImply/data/data_raw/data_itc99/bench',  '/home/zenu/code/FaultPreImply/data/data_raw/data_itc99/fault_mesg']\
            ,['/home/zenu/code/FaultPreImply/data/data_raw/data_epfl/bench',   '/home/zenu/code/FaultPreImply/data/data_raw/data_epfl/fault_mesg']\
            ]

data_folds_test = [ \
                    # ['/home/zenu/code/FaultPreImply/data/data_raw/data_iscas85/bench','/home/zenu/code/FaultPreImply/data/data_raw/data_iscas85/fault_mesg']\
                    ['/home/zenu/code/FaultPreImply/data/data_raw/data_test1/bench','/home/zenu/code/FaultPreImply/data/data_raw/data_test1/fault_mesg']\
                    ,['/home/zenu/code/FaultPreImply/data/data_raw/data_test2/bench','/home/zenu/code/FaultPreImply/data/data_raw/data_test2/fault_mesg']\
                    ]
NO_PATTERNS = 15000

gate_to_index = {'INPUT': 0, 'AND': 1, 'NOT': 2}
MIN_LEVEL = 3
MIN_PI_SIZE = 4
MAX_INCLUDE = 1.5
MAX_PROB_GAP = 0.05
MAX_LEVEL_GAP = 5

MIDDLE_DIST_IGNORE = [0.2, 0.8]

def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', default='train')
    parser.add_argument('--start_idx', default=0, type=int)
    parser.add_argument('--end_idx', default=10000, type=int)
    parser.add_argument('--aig_folder', default='./data/test')

    args = parser.parse_args()
    return args

def gen_tt_pair(x_data, fanin_list, fanout_list, level_list, tt_prob):
    tt_len = len(tt[0])
    pi_cone_list = []
    for idx in range(len(x_data)):
        pi_cone_list.append([])

    # Get pre fanout
    for level in range(len(level_list)):
        if level == 0:
            for idx in level_list[level]:
                pi_cone_list[idx].append(idx)
        else:
            for idx in level_list[level]:
                for fanin_idx in fanin_list[idx]:
                    pi_cone_list[idx] += pi_cone_list[fanin_idx]
                pre_dist = Counter(pi_cone_list[idx])
                pi_cone_list[idx] = list(pre_dist.keys())

    # Pair
    tt_pair_index = []
    tt_dis = []
    min_tt_dis = []
    for i in range(len(x_data)):
        if x_data[i][2] < MIN_LEVEL or len(pi_cone_list[i]) < MIN_PI_SIZE:
            continue
        for j in range(i+1, len(x_data), 1):
            if x_data[j][2] < MIN_LEVEL or len(pi_cone_list[j]) < MIN_PI_SIZE:
                continue
            # Cond. 2: probability
            if abs(tt_prob[i] - tt_prob[j]) > MAX_PROB_GAP:
                continue
            # Cond. 1: Level
            if abs(x_data[i][2] - x_data[j][2]) > MAX_LEVEL_GAP:
                continue

            # Cond. 5: Include
            if pi_cone_list[i] != pi_cone_list[j]:
                continue

            distance = np.array(tt[i]) - np.array(tt[j])
            distance_value = np.linalg.norm(distance, ord=1) / tt_len

            # Cond. 4: Extreme distance
            if distance_value > MIDDLE_DIST_IGNORE[0] and distance_value < MIDDLE_DIST_IGNORE[1]:
                continue
            
            # pair idx
            tt_pair_index.append([i, j])
            # pair true table normalized distance
            tt_dis.append(distance_value)
            # pair true table normalized negative distance 
            distance_e = (1-np.array(tt[i])) - np.array(tt[j])
            # minimum of distance and distance_e
            min_distance = min(np.linalg.norm(distance, ord=1), np.linalg.norm(distance_e, ord=1))
            min_tt_dis.append(min_distance / tt_len)

    return tt_pair_index, tt_dis, min_tt_dis

def get_redundant_mesg(x_data,circuit2fault_mesg):
    UR_fault = [[0]] * len(x_data)
    for idx,x in enumerate(x_data):
        assert(x[0] in circuit2fault_mesg)
        for fault_class_mesg in circuit2fault_mesg[x[0]]:
            if fault_class_mesg[2] == "UR":
                UR_fault[idx] = [1]
                break
    return UR_fault
        
        
def parse_fault_mesg(bench_filename):
    circuit2fault_mesg = dict()
    bench_filename_list = bench_filename.split('/')
    bench_filename_list[-2] = 'fault_mesg'
    bench_filename_list[-1] = '_'.join(bench_filename_list[-1].split('_')[:-1])
    bench_filename_list[-1] = bench_filename_list[-1]+'_wire2fault[after.tmax.flist].label'
    fault_mesg_filename = '/'.join(bench_filename_list)
    is_wire_line = True
    wire2fault_mesg = ""
    with open(fault_mesg_filename, mode='r') as fin:
        for line in fin.readlines():
            if is_wire_line:
                is_wire_line = False
                wire2fault_mesg = line
            else:
                is_wire_line = True
                wire2fault_mesg=wire2fault_mesg+line
                wire2fault_mesg = wire2fault_mesg.replace('\n','')
                fault_mesgs = wire2fault_mesg.split(' ')
                
                wire_name = fault_mesgs[0]
                wire_fault_num = int(fault_mesgs[1])
                # print(fault_mesgs)
                circuit2fault_mesg[wire_name] = []
                fault_mesgs = fault_mesgs[2:]
                for i in range(wire_fault_num):
                    # print([fault_mesgs[3*i],fault_mesgs[3*i+1],fault_mesgs[3*i+2]])
                    circuit2fault_mesg[wire_name].append([fault_mesgs[3*i],fault_mesgs[3*i+1],fault_mesgs[3*i+2]])
        fin.close()
    # print(circuit2fault_mesg)
    return circuit2fault_mesg
                
if __name__ == '__main__':
    graphs = {}
    labels = {}
    args = get_parse_args()
    output_folder = './data/data_npz/{}'.format(args.exp_id)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    tot_circuit = 0
    cir_idx = 0
    tot_nodes = 0
    tot_pairs = 0
    for data_path,fault_mesg_path in data_folds :
        name_list = []
        aig_folder = data_path
        print('[INFO] Read bench from: ', aig_folder)
        for bench_filename in glob.glob(os.path.join(aig_folder, '*.bench')):
            tot_circuit += 1
            name_list.append(bench_filename)
        for bench_filename in name_list[args.start_idx: min(args.end_idx, len(name_list))]:
            circuit_name = bench_filename.split('/')[-1].split('.')[0]
            bench_name = bench_filename.split('/')[-3]
            simple_circuit_name = '_'.join(circuit_name.split('_')[:-1])
            # print(bench_name,simple_circuit_name)
            #-------------------------------parse circuir netlist#-------------------------------
            x_data_old_name, edge_index, fanin_list, fanout_list, level_list = circuit_utils.parse_bench_with_old_name(bench_filename, gate_to_index)
            # print(x_data_old_name)
            # print(simple_circuit_name)
            circuit2fault_mesg = parse_fault_mesg(bench_filename)
            
            #-------------------------------redundant fault mesg #-------------------------------
            has_redundant_fault = get_redundant_mesg(x_data_old_name,circuit2fault_mesg)
            # print(has_redundant_fault)
            #-------------------------------rename node to idx #-------------------------------
            x_data = utils.rename_node(x_data_old_name)
            
            #-------------------------------Simulation #-------------------------------
            start_time = time.time()
            PI_index = level_list[0]
            if len(PI_index) < 13:
                tt = circuit_utils.simulator_truth_table(x_data, PI_index, level_list, fanin_list, gate_to_index)
            else:
                tt = circuit_utils.simulator_truth_table_random(x_data, PI_index, level_list, fanin_list, gate_to_index, NO_PATTERNS)
            y = [0] * len(x_data)
            for idx in range(len(x_data)):
                y[idx] = np.sum(tt[idx]) / len(tt[idx])
                
            #-------------------------------Get Pair #-------------------------------
            tt_pair_index, tt_dis, min_tt_dis = gen_tt_pair(x_data, fanin_list, fanout_list, level_list, y)
            end_time = time.time()
            
            #-------------------------------x_data[0]:gate_id  x_data[1]:gate_type  x_data[2]:gate_dpi
            graphs[circuit_name] = {'x': np.array(x_data).astype('float32'), "edge_index": np.array(edge_index)}
            labels[circuit_name] = {
                'tt_pair_index': np.array(tt_pair_index), 'tt_dis': np.array(tt_dis).astype('float32'), 
                'prob': np.array(y).astype('float32'), 
                'min_tt_dis': np.array(min_tt_dis).astype('float32'), 
                'has_redundant_fault':np.array(has_redundant_fault).astype('float32'), 
            }
            
            
            tot_nodes += len(x_data)
            tot_pairs += len(tt_dis)
            
            print('\033[32mSave: {}, # PI: {:}, Tot Pairs: {}, time: {:.2f} s ({:} / {:})\033[0m'.format(
                circuit_name, len(PI_index), tot_pairs, end_time - start_time, cir_idx, args.end_idx - args.start_idx
            ))

            if cir_idx != 0 and cir_idx % 1000 == 0:
                output_filename_circuit = os.path.join(output_folder, 'tmp_{:}_graphs.npz'.format(cir_idx))
                output_filename_labels = os.path.join(output_folder, 'tmp_{:}_labels.npz'.format(cir_idx))
                np.savez_compressed(output_filename_circuit, circuits=graphs)
                np.savez_compressed(output_filename_labels, labels=labels)
            cir_idx += 1

    output_filename_circuit = os.path.join(output_folder, 'graphs.npz')
    output_filename_labels = os.path.join(output_folder, 'labels.npz')
    print('# Graphs: {:}, # Nodes: {:}'.format(len(graphs), tot_nodes))
    print('Total pairs: ', tot_pairs)
    np.savez_compressed(output_filename_circuit, circuits=graphs)
    np.savez_compressed(output_filename_labels, labels=labels)
    print(output_filename_circuit)
