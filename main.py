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
from tinyturbo import train, test, TinyTurbo
from utils import moving_average

def get_args():
    parser = argparse.ArgumentParser(description='TinyTurbo')

    # Turbo
    parser.add_argument('--block_len', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--turbo_iters', type=int, default=6)
    parser.add_argument('--decoding_type', type=str, choices=['normal', 'normal_common', 'same_all', 'same_iteration', 'scale', 'scale_common', 'same_scale', 'same_scale_iteration', 'one_weight'], default='scale')
    parser.add_argument('--interleaver', type=str, choices=['random', 'qpp', 'rectangular'], default='qpp')
    parser.add_argument('--puncture', dest = 'puncture', default=False, action='store_true', help='Puncture to get rate 1/2')
    parser.add_argument('--code', type=str, choices=['lte', '757'], default='lte', help = 'Turbo code to use')

    # TinyTurbo
    parser.add_argument('--target', type=str, choices=['gt', 'LLR'], default='gt', help = 'Train with ground truth, or with LLRs of Turbo decoder?')
    parser.add_argument('--init_type', type=str, choices=['ones', 'random', 'gaussian'], default='ones')
    parser.add_argument('--num_steps', type=int, default=5000)
    parser.add_argument('--tinyturbo_iters', type=int, default=3)
    parser.add_argument('--tt_bcjr', type=str, choices=['MAP', 'max_log_MAP'], default='max_log_MAP')
    parser.add_argument('--train_snr', type=float, default=-1)

    parser.add_argument('--input', type=str, choices=['y', 'llr'], default='llr', help = 'Input to tinyturbo: y or LLR?')
    parser.add_argument('--loss_type', type=str, choices=['BCE', 'MSE'], default='BCE', help = 'Train loss')
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--save_every', type=int, default=100)

    # Testing
    parser.add_argument('--test_size', type=int, default=10000)
    parser.add_argument('--test_batch_size', type=int, default=10000)
    parser.add_argument('--snr_points', type=int, default=8)
    parser.add_argument('--test_snr_start', type=float, default=-1.5)
    parser.add_argument('--test_snr_end', type=float, default=2)
    parser.add_argument('--test_block_len', type=int, default=None)

    parser.add_argument('--noise_type', type=str, choices=['awgn', 'fading', 'radar', 't-dist', 'EPA', 'EVA', 'ETU', 'MIMO'], default='awgn')
    parser.add_argument('--vv',type=float, default=5, help ='only for t distribution channel : degrees of freedom')
    parser.add_argument('--radar_prob',type=float, default=0.01, help ='only for radar distribution channel')
    parser.add_argument('--radar_power',type=float, default=5.0, help ='only for radar distribution channel')

    parser.add_argument('--gpu', type=int, default=0) #-1 for cpu
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--inter_seed', type=int, default=0)
    parser.add_argument('--id', type=str, default=None)
    parser.add_argument('--test', dest = 'test', default=False, action='store_true', help='Testing?')
    parser.add_argument('--test_all', dest = 'test_all', default=False, action='store_true', help='Testing?')

    parser.add_argument('--load_model_train', type=str, default=None, help='load model to initialize training')
    parser.add_argument('--load_model_step', type=int, default=None, help='test at model saved after x steps')

    # Real data?
    parser.add_argument('--only_args', dest = 'only_args', default=False, action='store_true')
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

    # torch.autograd.set_detect_anomaly(True)

    if args.id is None:
        args.id = str(np.random.randint(10000, 99999))
    args.save_path = os.path.join('Results', args.id)

    # torch.manual_seed(args.seed)

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

    if args.interleaver == 'qpp':
        if args.block_len == 40:
            p_array = get_qpp(3, 10, 40)
            interleaver.set_p_array(p_array)
        elif args.block_len == 64:
            p_array = get_qpp(7, 16, 64)
            interleaver.set_p_array(p_array)
        elif args.block_len == 104:
            p_array = get_qpp(7, 26, 104)
            interleaver.set_p_array(p_array)
        elif args.block_len == 200:
            p_array = get_qpp(13, 50, 200)
            interleaver.set_p_array(p_array)
        elif args.block_len == 504:
            p_array = get_qpp(55,84, 504)
            interleaver.set_p_array(p_array)
        elif args.block_len == 1008:
            p_array = get_qpp(55, 84, 1008)
            interleaver.set_p_array(p_array)
        else:
            print("QPP not yet supported for block length {}".format(args.block_len))
    # interleaver = Interleaver(args.block_len, 0, p_array) # for custom permutation
    print('Using interleaver p-array : {}'.format(list(interleaver.p_array)))

    if args.only_args:
        print("Loaded args. Exiting")
        sys.exit()

    # Training

    if not args.test:
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
        loaded_weights = None
        if args.load_model_train is not None:
            try:
                checkpoint = torch.load(args.load_model_train)
                loaded_weights = checkpoint['weights']
                print("Training init from model at {}".format(args.load_model_train))
            except:
                print("Model not found at {}".format(args.load_model_train))

        tinyturbo, training_losses, training_bers, steps = train(args, trellis1, trellis2, interleaver, device, loaded_weights)

        torch.save({'weights': tinyturbo.cpu().state_dict(), 'args': args, 'steps': steps, 'p_array':interleaver.p_array}, os.path.join(args.save_path, 'models/weights.pt'))
        torch.save({'weights': tinyturbo.cpu().state_dict(), 'args': args, 'steps': steps, 'p_array':interleaver.p_array}, os.path.join(args.save_path, 'models/weights_{}.pt'.format(int(steps))))
        plt.figure()
        plt.plot(training_losses)
        plt.plot(moving_average(training_losses, n=10))
        plt.savefig(os.path.join(args.save_path, 'training_losses.png'))
        plt.close()

        plt.figure()
        plt.plot(training_losses)
        plt.plot(moving_average(training_losses, n=10))
        plt.yscale('log')
        plt.savefig(os.path.join(args.save_path, 'training_losses_log.png'))
        plt.close()

        plt.figure()
        plt.plot(training_bers)
        plt.plot(moving_average(training_bers, n=10))
        plt.savefig(os.path.join(args.save_path, 'training_bers.png'))
        plt.close()

        plt.figure()
        plt.plot(training_bers)
        plt.plot(moving_average(training_bers, n=10))
        plt.yscale('log')
        plt.savefig(os.path.join(args.save_path, 'training_bers_log.png'))
        plt.close()


    if args.test and args.load_model_step is not None:
        checkpoint = torch.load(os.path.join(args.save_path, 'models/weights_{}.pt'.format(args.load_model_step)), map_location = device)
    else:
        checkpoint = torch.load(os.path.join(args.save_path, 'models/weights.pt'), map_location=device)
    trained_args = checkpoint['args']
    weights = checkpoint['weights']

    if args.test:
        print("Loaded model at step {}".format(checkpoint['steps']))
        interleaver.set_p_array(checkpoint['p_array'])

    if args.test_block_len is not None:
        args.block_len = args.test_block_len
        if args.block_len == 40:
            p_array = get_qpp(3, 10, 40)
            interleaver.set_p_array(p_array)
        elif args.block_len == 64:
            p_array = get_qpp(7, 16, 64)
            interleaver.set_p_array(p_array)
        elif args.block_len == 104:
            p_array = get_qpp(7, 26, 104)
            interleaver.set_p_array(p_array)
        elif args.block_len == 200:
            p_array = get_qpp(13, 50, 200)
            interleaver.set_p_array(p_array)
        elif args.block_len == 504:
            p_array = get_qpp(55,84, 504)
            interleaver.set_p_array(p_array)
        elif args.block_len == 1008:
            p_array = get_qpp(55, 84, 1008)
            interleaver.set_p_array(p_array)
        else:
            print("QPP not yet supported for block length {}".format(args.block_len))
        print('Testing block_len = {}'.format(args.test_block_len))
    else:
        args.test_block_len = args.block_len

    tinyturbo = TinyTurbo(args.test_block_len, args.tinyturbo_iters, device, args.init_type, args.decoding_type)
    tinyturbo.load_state_dict(weights)
    tinyturbo.to(device)
    if args.only_tt:
        snr_range, bers_ml, bers_l, bers_tt, blers_ml, blers_l, blers_tt = test(args, tinyturbo, trellis1, trellis2, interleaver, device, only_tt = True)
    else:
        snr_range, bers_ml, bers_l, bers_tt, blers_ml, blers_l, blers_tt = test(args, tinyturbo, trellis1, trellis2, interleaver, device, only_tt = False)
        snr_range_saved = snr_range
    print('SNRs = {}'.format(snr_range))
    print("BERs : \n Max-Log-MAP : {}, \n MAP = {}, \n TinyTurbo = {}\n".format(bers_ml, bers_l, bers_tt))
    print("BLERs : \n Max-Log-MAP : {}, \n MAP = {}, \n TinyTurbo = {}\n".format(blers_ml, blers_l, blers_tt))

    if not args.only_tt:
        plt.figure()
        plt.semilogy(snr_range_saved, bers_ml, marker = '*', linewidth=1.5, label = 'Max-Log-MAP it={}'.format(args.tinyturbo_iters))
        plt.semilogy(snr_range_saved, bers_l, marker = 'v', linewidth=1.5, label = 'MAP it={}'.format(args.turbo_iters))
        plt.semilogy(snr_range, bers_tt, marker = '^', linewidth=1.5, label = 'TinyTurbo it={}'.format(args.tinyturbo_iters))

        plt.grid()
        plt.xlabel("SNR (dB)", fontsize=16)
        plt.ylabel("Bit Error Rate", fontsize=16)
        plt.title("TinyTurbo block_len = {}, trained at {}dB".format(args.block_len, trained_args.train_snr))
        plt.legend(loc = 'best')
        plt.savefig(os.path.join(args.save_path, 'ber_step_{}_{}.pdf'.format(checkpoint['steps'], args.noise_type)))
        plt.close()

        plt.figure()
        plt.semilogy(snr_range_saved, blers_ml, marker = 'o', linewidth=1.5, label = 'Max-Log-MAP it={}'.format(args.tinyturbo_iters))
        plt.semilogy(snr_range_saved, blers_l, marker = 'P', linewidth=1.5, label = 'MAP it={}'.format(args.turbo_iters))
        plt.semilogy(snr_range, blers_tt, marker = '^', linewidth=1.5, label = 'TinyTurbo it={}'.format(args.tinyturbo_iters))

        plt.grid()
        plt.xlabel("SNR (dB)", fontsize=16)
        plt.ylabel("Block Error Rate", fontsize=16)
        plt.title("TinyTurbo block_len = {}, trained at {}dB".format(args.block_len, trained_args.train_snr))
        plt.legend(loc = 'best')
        plt.savefig(os.path.join(args.save_path, 'bler_step_{}_{}.pdf'.format(checkpoint['steps'], args.noise_type)))
        plt.close()

    print("Saved at: {}".format(args.save_path))
