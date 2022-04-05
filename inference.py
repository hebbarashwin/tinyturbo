import torch
import torch.utils.data as data
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time
import pickle

from convcode import Trellis
from interleaver import Interleaver
from tinyturbo import tinyturbo_decode
from turbo import turbo_encode, turbo_decode
from utils import moving_average, snr_db2sigma, errors_ber, errors_bler, corrupt_signal

def get_args():
    parser = argparse.ArgumentParser(description='TinyTurbo')

    # Turbo
    parser.add_argument('--block_len', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--turbo_iters', type=int, default=6)
    parser.add_argument('--bcjr_method', type=str, choices=['MAP', 'max_log_MAP'], default='MAP')
    parser.add_argument('--decoding_type', type=str, choices=['normal', 'normal_common', 'same_all', 'same_iteration', 'scale', 'scale_common', 'same_scale', 'same_scale_iteration', 'one_weight'], default='scale')
    parser.add_argument('--interleaver', type=str, choices=['random', 'qpp', 'rectangular'], default='qpp')
    parser.add_argument('--puncture', dest = 'puncture', default=False, action='store_true', help='Puncture to get rate 1/2')
    parser.add_argument('--code', type=str, choices=['lte', '757'], default='lte', help = 'Turbo code to use')
    parser.add_argument('--tt_bcjr', type=str, choices=['MAP', 'max_log_MAP'], default='max_log_MAP')

    # TinyTurbo
    parser.add_argument('--tinyturbo_iters', type=int, default=3)

    # Testing
    parser.add_argument('--test_size', type=int, default=100000)
    parser.add_argument('--test_batch_size', type=int, default=100000)
    parser.add_argument('--snr_points', type=int, default=8)
    parser.add_argument('--test_snr_start', type=float, default=-8)
    parser.add_argument('--test_snr_end', type=float, default=0)
    parser.add_argument('--test_block_len', type=int, default=None)

    parser.add_argument('--noise_type', type=str, choices=['awgn', 'fading', 'radar', 't-dist', 'EPA', 'EVA', 'ETU', 'MIMO'], default='awgn')
    parser.add_argument('--vv',type=float, default=5, help ='only for t distribution channel : degrees of freedom')
    parser.add_argument('--radar_prob',type=float, default=0.01, help ='only for radar distribution channel')
    parser.add_argument('--radar_power',type=float, default=5.0, help ='only for radar distribution channel')

    parser.add_argument('--gpu', type=int, default=-1) #-1 for cpu
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--inter_seed', type=int, default=0)
    parser.add_argument('--id', type=str, default=None)
    parser.add_argument('--test_all', dest = 'test_all', default=False, action='store_true', help='Testing?')

    parser.add_argument('--load_model_step', type=int, default=None, help='test at model saved after x steps')
    parser.add_argument('--only_tt', dest = 'only_tt', default=False, action='store_true')

    args = parser.parse_args()

    return args

def get_qpp(f1, f2, block_len):
    nums = np.arange(block_len)
    inds = (f1*nums + f2*(nums**2))%block_len

    return inds

if __name__ == '__main__':

    args = get_args()
    print(args)

    if args.gpu == -1:
        device = torch.device('cpu')
        print("USING CPU")
    else:
        device = torch.device("cuda:{0}".format(args.gpu))
        print("USING GPU {}".format(args.gpu))


    if args.id is None:
        args.id = str(np.random.randint(10000, 99999))
    args.save_path = os.path.join('Results', args.id)

    if args.code == '757':
        # Turbo-757 parameters
        M = np.array([2])                         # Number of delay elements in the convolutional encoder
        generator_matrix = np.array([[7, 5]])     # Encoder of convolutional encoder
        feedback = 7
    else:
        # Turbo-LTE parameters
        M = np.array([3])                         # Number of delay elements in the convolutional encoder
        generator_matrix = np.array([[11, 13]])     # Encoder of convolutional encoder
        feedback = 11

    trellis1 = Trellis(M, generator_matrix, feedback, 'rsc')
    trellis2 = Trellis(M, generator_matrix, feedback, 'rsc')
    interleaver = Interleaver(args.block_len, args.inter_seed)

    # Loading model
    if args.load_model_step is not None:
        checkpoint = torch.load(os.path.join(args.save_path, 'models/weights_{}.pt'.format(args.load_model_step)), map_location = device)
    else:
        checkpoint = torch.load(os.path.join(args.save_path, 'models/weights.pt'), map_location=device)
    trained_args = checkpoint['args']
    weight_d = checkpoint['weights']
    for ii in range(args.tinyturbo_iters):
        weight_d['normal'][ii].to(device)
        weight_d['interleaved'][ii].to(device)

    print("Loaded model at step {}".format(checkpoint['steps']))
    interleaver.set_p_array(checkpoint['p_array'])
    print('Using interleaver p-array : {}'.format(list(interleaver.p_array)))

    # if args.only_tt:
    #     snr_range, bers_ml, bers_l, bers_tt, blers_ml, blers_l, blers_tt = test(args, weight_d, trellis1, trellis2, interleaver, device, only_tt = True)
    # else:
    #     snr_range, bers_ml, bers_l, bers_tt, blers_ml, blers_l, blers_tt = test(args, weight_d, trellis1, trellis2, interleaver, device, only_tt = False)
    #     snr_range_saved = snr_range
    # print('SNRs = {}'.format(snr_range))
    # print("BERs : \n Max-Log-MAP : {}, \n MAP = {}, \n TinyTurbo = {}\n".format(bers_ml, bers_l, bers_tt))
    # print("BLERs : \n Max-Log-MAP : {}, \n MAP = {}, \n TinyTurbo = {}\n".format(blers_ml, blers_l, blers_tt))



    if args.snr_points == 1 and args.test_snr_start == args.test_snr_end:
        snr_range = [args.test_snr_start]
    else:
        snrs_interval = (args.test_snr_end - args.test_snr_start) * 1.0 / (args.snr_points-1)
        snr_range = [snrs_interval * item + args.test_snr_start for item in range(args.snr_points)]

    num_batches = args.test_size // args.test_batch_size
    noise_type = args.noise_type

    bers_ml = []
    blers_ml = []
    bers_l = []
    blers_l = []
    bers_tt = []
    blers_tt = []
    print("TESTING")

    if  args.noise_type in ['EPA', 'EVA', 'ETU']: #run from MATLAB
        print("Using ", args.noise_type, " channel")
        import matlab.engine
        eng = matlab.engine.start_matlab()
        s = eng.genpath('matlab_scripts')
        eng.addpath(s, nargout=0)
        message_bits = torch.randint(0, 2, (args.test_batch_size, args.block_len), dtype=torch.float).to(device)
        coded = turbo_encode(message_bits, trellis1, trellis2, interleaver, puncture = args.puncture).to(device)
        coded_mat = matlab.double(coded.numpy().tolist())
        # calculate closest multiple to num_sym(179)
        num_sym = int(np.floor(coded.size(0)/179)) + 1
        code_len = int((args.block_len*3)+4*(trellis1.total_memory))
        num_blocks = 179
        SNRs = matlab.double(snr_range)
        rx_llrs = eng.generate_lte_data(coded_mat, code_len, args.noise_type, SNRs, num_blocks, num_sym)
        # convert to numpy
        rx_llrs = np.array(rx_llrs)
        eng.quit()
    elif args.noise_type == 'MIMO':
        print("Using ", args.noise_type, " channel")
        import matlab.engine
        eng = matlab.engine.start_matlab()
        s = eng.genpath('matlab_scripts')
        eng.addpath(s, nargout=0)
        message_bits = torch.randint(0, 2, (args.test_batch_size, args.block_len), dtype=torch.float).to(device)
        coded = turbo_encode(message_bits, trellis1, trellis2, interleaver, puncture = args.puncture).to(device)
        coded_mat = matlab.double(coded.numpy().tolist())
        code_len = int((args.block_len*3)+4*(trellis1.total_memory))
        num_blocks = 179
        SNRs = matlab.double(snr_range)
        num_tx = 1
        num_rx = 2
        max_num_tx = 2
        max_num_rx = 2
        num_codewords = int(args.test_size)
        rx_llrs =  eng.generate_mimo_diversity_data (num_tx, num_rx, max_num_tx, max_num_rx, coded_mat, code_len, SNRs, num_codewords)
        # convert to numpy
        rx_llrs = np.array(rx_llrs)
        eng.quit()

    for ii in range(num_batches):
        for k, snr in enumerate(snr_range):
            if args.noise_type in ['awgn', 'fading', 't-dist', 'radar']:
                sigma = snr_db2sigma(snr)
                noise_variance = sigma**2
                message_bits = torch.randint(0, 2, (args.test_batch_size, args.block_len), dtype=torch.float).to(device)
                coded = turbo_encode(message_bits, trellis1, trellis2, interleaver, puncture = args.puncture).to(device)
                noisy_coded = corrupt_signal(coded, sigma, noise_type, vv = args.vv, radar_power = args.radar_power, radar_prob = args.radar_prob)
                received_llrs = 2*noisy_coded/noise_variance

            elif noise_type in ['EPA', 'EVA', 'ETU', 'MIMO']:
                # converting numpy to torch here
                received_llrs = torch.from_numpy(np.transpose(rx_llrs[:, :, k])).to(device)

            if not args.only_tt:
                # Turbo decode
                ml_llrs, decoded_ml = turbo_decode(received_llrs, trellis1, args.tinyturbo_iters,
                                             interleaver, method='max_log_MAP', puncture = args.puncture)
                ber_maxlog = errors_ber(message_bits, decoded_ml[:, :-trellis1.total_memory])
                bler_maxlog = errors_bler(message_bits, decoded_ml[:, :-trellis1.total_memory])

                if ii == 0:
                    bers_ml.append(ber_maxlog/num_batches)
                    blers_ml.append(bler_maxlog/num_batches)
                else:
                    bers_ml[k] += ber_maxlog/num_batches
                    blers_ml[k] += bler_maxlog/num_batches

                l_llrs, decoded_l = turbo_decode(received_llrs, trellis1, args.turbo_iters,
                                            interleaver, method='log_MAP', puncture = args.puncture)
                ber_log = errors_ber(message_bits, decoded_l[:, :-trellis1.total_memory])
                bler_log = errors_bler(message_bits, decoded_l[:, :-trellis1.total_memory])

                if ii == 0:
                    bers_l.append(ber_log/num_batches)
                    blers_l.append(bler_log/num_batches)
                else:
                    bers_l[k] += ber_log/num_batches
                    blers_l[k] += bler_log/num_batches

            # tinyturbo decode
            tt_llrs, decoded_tt = tinyturbo_decode(weight_d, received_llrs, trellis1, args.tinyturbo_iters, interleaver, method = args.tt_bcjr, puncture = args.puncture)
            
            ber_tinyturbo = errors_ber(message_bits, decoded_tt[:, :-trellis1.total_memory])
            bler_tinyturbo = errors_bler(message_bits, decoded_tt[:, :-trellis1.total_memory])

            if ii == 0:
                bers_tt.append(ber_tinyturbo/num_batches)
                blers_tt.append(bler_tinyturbo/num_batches)
            else:
                bers_tt[k] += ber_tinyturbo/num_batches
                blers_tt[k] += bler_tinyturbo/num_batches

    print('SNRs = {}'.format(snr_range))
    print("BERs : \n Max-Log-MAP : {}, \n MAP = {}, \n TinyTurbo = {}\n".format(bers_ml, bers_l, bers_tt))
    print("BLERs : \n Max-Log-MAP : {}, \n MAP = {}, \n TinyTurbo = {}\n".format(blers_ml, blers_l, blers_tt))