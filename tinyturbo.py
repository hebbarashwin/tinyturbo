__author__ = 'hebbarashwin'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import csv
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from tqdm import tqdm

from turbo import turbo_encode, turbo_decode, bcjr_decode
from utils import snr_db2sigma, errors_ber, errors_bler, corrupt_signal, moving_average

class Turbo_subnet(nn.Module):
    def __init__(self, block_len, init_type = 'ones', one_weight = False):
        super(Turbo_subnet, self).__init__()

        assert init_type in ['ones', 'random', 'gaussian'], "Invalid init type"
        if init_type == 'ones':
            self.w1 = nn.parameter.Parameter(torch.ones((1, block_len)))
            self.w2 = nn.parameter.Parameter(torch.ones((1, block_len)))
            self.w3 = nn.parameter.Parameter(torch.ones((1, block_len)))
        elif init_type == 'random':
            self.w1 = nn.parameter.Parameter(torch.rand((1, block_len)))
            self.w2 = nn.parameter.Parameter(torch.rand((1, block_len)))
            self.w3 = nn.parameter.Parameter(torch.rand((1, block_len)))
        elif init_type == 'gaussian':
            self.w1 = nn.parameter.Parameter(0.001* torch.randn((1, block_len)))
            self.w2 = nn.parameter.Parameter(0.001*torch.randn((1, block_len)))
            self.w3 = nn.parameter.Parameter(0.001*torch.randn((1, block_len)))

        if one_weight:
            self.w3 = self.w1
            self.w2 = self.w1

    def forward(self, L_ext, L_sys, L_int):

        x = self.w1 * L_ext - self.w2 * L_sys - self.w3 * L_int

        return x

def init_weights(block_len, num_iter, device = torch.device('cpu'), init_type = 'ones', type = 'normal'):
    """
    Initialize weights for TinyTurbo
    Weight entanglement described in paper: 'scale'

    Other settings are ablation studies.
    """
    weight_dict = {}
    normal = {}
    interleaved = {}

    assert type in ['normal', 'normal_common', 'same_all', 'same_iteration', 'scale', 'scale_common', 'same_scale_iteration', 'same_scale', 'one_weight']

    if type == 'normal':
        for ii in range(num_iter):
            normal[ii] = Turbo_subnet(block_len, init_type).to(device)
            interleaved[ii] = Turbo_subnet(block_len, init_type).to(device)
        weight_dict['normal'] = normal
        weight_dict['interleaved'] = interleaved

    if type == 'normal_common':
        for ii in range(num_iter):
            net = Turbo_subnet(block_len, init_type).to(device)
            normal[ii] = net
            interleaved[ii] = net
        weight_dict['normal'] = normal
        weight_dict['interleaved'] = interleaved

    elif type == 'same_all':
        net = Turbo_subnet(block_len, init_type).to(device)
        for ii in range(num_iter):
            normal[ii] = net
            interleaved[ii] = net
        weight_dict['normal'] = normal
        weight_dict['interleaved'] = interleaved

    elif type == 'same_iteration':
        normal_net = Turbo_subnet(block_len, init_type).to(device)
        interleaved_net = Turbo_subnet(block_len, init_type).to(device)

        for ii in range(num_iter):
            normal[ii] = normal_net
            interleaved[ii] = interleaved_net
        weight_dict['normal'] = normal
        weight_dict['interleaved'] = interleaved

    elif type == 'scale':
        for ii in range(num_iter):
            normal[ii] = Turbo_subnet(1, init_type).to(device)
            interleaved[ii] = Turbo_subnet(1, init_type).to(device)
        weight_dict['normal'] = normal
        weight_dict['interleaved'] = interleaved

    elif type == 'scale_common':
        for ii in range(num_iter):
            net = Turbo_subnet(1, init_type).to(device)
            normal[ii] = net
            interleaved[ii] = net
        weight_dict['normal'] = normal
        weight_dict['interleaved'] = interleaved

    elif type == 'same_scale':
        net = Turbo_subnet(1, init_type).to(device)
        for ii in range(num_iter):
            normal[ii] = net
            interleaved[ii] = net
        weight_dict['normal'] = normal
        weight_dict['interleaved'] = interleaved

    elif type == 'same_scale_iteration':
        net_normal = Turbo_subnet(1, init_type).to(device)
        net_interleaved = Turbo_subnet(1, init_type).to(device)
        for ii in range(num_iter):
            normal[ii] = net_normal
            interleaved[ii] = net_interleaved
        weight_dict['normal'] = normal
        weight_dict['interleaved'] = interleaved

    elif type == 'one_weight':
        net = Turbo_subnet(1, init_type, one_weight = True).to(device)
        for ii in range(num_iter):
            normal[ii] = net
            interleaved[ii] = net
        weight_dict['normal'] = normal
        weight_dict['interleaved'] = interleaved

    return weight_dict

def tinyturbo_decode(weight_dict, received_llrs, trellis, number_iterations, interleaver, L_int = None, method = 'max_log_MAP', puncture = False):

    """ Turbo Decoder.
    Decode a Turbo code using TinyTurbo weights.

    Parameters
    ----------
    weight_dict : Dictionary
        Contains normal and interleaved weights for TinyTurbo
    received_llrs : LLRs of shape (batch_size, 3*M + 4*memory)
        Received LLRs corresponding to the received Turbo encoded bits
    trellis : Trellis object
        Trellis representation of the convolutional code
    number_iterations: Int
        Number of iterations of BCJR algorithm
    interleaver : Interleaver object
        Interleaver used in the turbo code.
    L_int : intrinsic LLRs of shape (batch_size, 3*M + 4*memory)
        Intrinsic LLRs (prior). (Set to zeros if no prior)
    method : Turbo decoding method
        max-log-MAP or MAP
    puncture: Bool
        Currently supports only puncturing pattern '110101'

    Returns
    -------
    L_ext : torch Tensor of decoded LLRs, of shape (batch_size, M + memory)

    decoded_bits: L_ext > 0

        Decoded beliefs
    """
    coded = received_llrs[:, :-4*trellis.total_memory]
    term = received_llrs[:, -4*trellis.total_memory:]
    if puncture:
        block_len = coded.shape[1]//2
        inds = torch.Tensor([1, 1, 0, 1, 0, 1]).repeat(block_len//2).byte()
        zero_inserted = torch.zeros(received_llrs.shape[0], 3*block_len, device = received_llrs.device)
        zero_inserted[:, inds] = coded
        coded = zero_inserted.float()
    sys_stream = coded[:, 0::3]
    non_sys_stream1 = coded[:, 1::3]
    non_sys_stream2 = coded[:, 2::3]

    term_sys1 = term[:, :2*trellis.total_memory][:, 0::2]
    term_nonsys1 = term[:, :2*trellis.total_memory][:, 1::2]
    term_sys2 = term[:, 2*trellis.total_memory:][:, 0::2]
    term_nonsys2 = term[:, 2*trellis.total_memory:][:, 1::2]

    sys_llrs = torch.cat((sys_stream, term_sys1), -1)
    non_sys_llrs1 = torch.cat((non_sys_stream1, term_nonsys1), -1)

    sys_stream_inter = interleaver.interleave(sys_stream)
    sys_llrs_inter = torch.cat((sys_stream_inter, term_sys2), -1)

    non_sys_llrs2 = torch.cat((non_sys_stream2, term_nonsys2), -1)
    sys_llr = sys_llrs

    if L_int is None:
        L_int = torch.zeros_like(sys_llrs).to(coded.device)

    L_int_1 = L_int

    for iteration in range(number_iterations):
        [L_ext_1, decoded] = bcjr_decode(sys_llrs, non_sys_llrs1, trellis, L_int_1, method=method)


        L_ext = L_ext_1 - L_int_1 - sys_llr
        L_e_1 = L_ext_1[:, :sys_stream.shape[1]]
        L_1 = L_int_1[:, :sys_stream.shape[1]]

        L_int_2 = weight_dict['normal'][iteration](L_e_1, sys_llr[:, :sys_stream.shape[1]], L_1)
        L_int_2 = interleaver.interleave(L_int_2)
        L_int_2 = torch.cat((L_int_2, torch.zeros_like(term_sys1)), -1)

        [L_ext_2, decoded] = bcjr_decode(sys_llrs_inter, non_sys_llrs2, trellis, L_int_2, method=method)

        L_e_2 = interleaver.deinterleave(L_ext_2[:, :sys_stream.shape[1]])
        L_2 = interleaver.deinterleave(L_int_2[:, :sys_stream.shape[1]])

        L_int_1 = weight_dict['interleaved'][iteration](L_e_2, sys_llr[:, :sys_stream.shape[1]], L_2)
        L_int_1 = torch.cat((L_int_1, torch.zeros_like(term_sys1)), -1)
    LLRs = L_ext + L_int_1 + sys_llr
    decoded_bits = (LLRs > 0).float()

    return LLRs, decoded_bits

def train(args, trellis1, trellis2, interleaver, device, loaded_weights = None):
    """
    Training function

    If args.target == 'LLR', then training proceeds like Turbonet+
    (Y. He, J. Zhang, S. Jin, C.-K. Wen, and G. Y. Li, “Model-driven dnn
    decoder for turbo codes: Design, simulation, and experimental results,”
    IEEE Transactions on Communications, vol. 68, no. 10, pp. 6127–6140)

    """
    if loaded_weights is None:
        weight_dict = init_weights(args.block_len, args.tinyturbo_iters, device, args.init_type, args.decoding_type)
    else:
        weight_dict = loaded_weights
        for ii in range(args.tinyturbo_iters):
            weight_dict['normal'][ii].to(device)
            weight_dict['interleaved'][ii].to(device)
    params = []
    for ii in range(args.tinyturbo_iters):
        params += list(weight_dict['normal'][ii].parameters())
        params += list(weight_dict['interleaved'][ii].parameters())

    criterion = nn.BCEWithLogitsLoss() if args.loss_type == 'BCE' else nn.MSELoss()
    optimizer = optim.Adam(params, lr = args.lr)

    sigma = snr_db2sigma(args.train_snr)
    noise_variance = sigma**2

    noise_type = args.noise_type #if args.noise_type is not 'isi' else 'isi_1'
    print("TRAINING")
    training_losses = []
    training_bers = []

    try:
        for step in range(args.num_steps):
            start = time.time()
            message_bits = torch.randint(0, 2, (args.batch_size, args.block_len), dtype=torch.float).to(device)
            coded = turbo_encode(message_bits, trellis1, trellis2, interleaver, puncture = args.puncture).to(device)
            noisy_coded = corrupt_signal(coded, sigma, noise_type, vv = args.vv, radar_power = args.radar_power, radar_prob = args.radar_prob)

            #tinyturbo decode

            received_llrs = 2*noisy_coded/noise_variance

            if args.input == 'y':
                tinyturbo_llr, decoded_tt = tinyturbo_decode(weight_dict, noisy_coded, trellis1, args.tinyturbo_iters, interleaver, method = args.tt_bcjr, puncture = args.puncture)
            else:
                tinyturbo_llr, decoded_tt = tinyturbo_decode(weight_dict, received_llrs, trellis1, args.tinyturbo_iters, interleaver, method = args.tt_bcjr, puncture = args.puncture)


            if args.target == 'LLR':
                #Turbo decode
                log_map_llr, _ = turbo_decode(received_llrs, trellis1, args.turbo_iters, interleaver, method='log_MAP', puncture = args.puncture)
                loss = criterion(tinyturbo_llr, log_map_llr)
            elif args.target == 'gt':
                if args.loss_type == 'BCE':
                    loss = criterion(tinyturbo_llr[:, :-trellis1.total_memory], message_bits)
                elif args.loss_type == 'MSE':
                    loss = criterion(torch.tanh(tinyturbo_llr[:, :-trellis1.total_memory]/2.), 2*message_bits-1)
            ber = errors_ber(message_bits, decoded_tt[:, :-trellis1.total_memory])

            training_losses.append(loss.item())
            training_bers.append(ber)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1)%10 == 0:
                print('Step : {}, Loss = {:.5f}, BER = {:.5f}, {:.2f} seconds, ID: {}'.format(step+1, loss, ber, time.time() - start, args.id))

            if (step+1)%args.save_every == 0 or step==0:
                torch.save({'weights': weight_dict, 'args': args, 'steps': step+1, 'p_array':interleaver.p_array}, os.path.join(args.save_path, 'models/weights.pt'))
                torch.save({'weights': weight_dict, 'args': args, 'steps': step+1, 'p_array':interleaver.p_array}, os.path.join(args.save_path, 'models/weights_{}.pt'.format(int(step+1))))
            if (step+1)%10 == 0:
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

                with open(os.path.join(args.save_path, 'values_training.csv'), 'w') as f:

                    # using csv.writer method from CSV package
                    write = csv.writer(f)

                    write.writerow(list(range(1, step+1)))
                    write.writerow(training_losses)
                    write.writerow(training_bers)

        return weight_dict, training_losses, training_bers, step+1

    except KeyboardInterrupt:
        print("Exited")

        torch.save({'weights': weight_dict, 'args': args, 'steps': step+1, 'p_array':interleaver.p_array}, os.path.join(args.save_path, 'models/weights.pt'))
        torch.save({'weights': weight_dict, 'args': args, 'steps': step+1, 'p_array':interleaver.p_array}, os.path.join(args.save_path, 'models/weights_{}.pt'.format(int(step+1))))
        with open(os.path.join(args.save_path, 'values_training.csv'), 'w') as f:

             # using csv.writer method from CSV package
             write = csv.writer(f)
             write.writerow(list(range(1, step+1)))
             write.writerow(training_losses)
             write.writerow(training_bers)

        return weight_dict, training_losses, training_bers, step+1

def test(args, weight_d, trellis1, trellis2, interleaver, device, only_tt = False):
    """
    Test function
    """
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

    if args.noise_type in ['EPA', 'EVA', 'ETU']: #run from MATLAB
        import matlab.engine
        eng = matlab.engine.start_matlab()
        s = eng.genpath('matlab_scripts')
        eng.addpath(s, nargout=0)
        #[msg_data, enc_data, llr_data] = generate_lte_data(coding_scheme, msg_len, code_len, chan, SNRs, num_blocks, num_sym);
        # can't we make num blocks 10000?
        msgs, codewords, rx_llrs = eng.generate_lte_data('Turbo', args.block_len, (args.block_len*3)+4*(trellis1.total_memory), args.noise_type, snr_range, 179, num_batches)
        #assuming above arrays in numpy
        eng.quit()
    elif args.noise_type == 'MIMO':
        import matlab.engine
        eng = matlab.engine.start_matlab()
        s = eng.genpath('matlab_scripts')
        eng.addpath(s, nargout=0)
        num_tx = 1
        num_rx = 2
        max_num_tx = 2
        max_num_rx = 2
        msgs, codewords, rx_llrs =  eng.generate_mimo_diversity_data (num_tx, num_rx, max_num_tx, max_num_rx, args.block_len, (args.block_len*3)+4*(trellis1.total_memory), snr_range, args.test_size)
        eng.quit()

    for ii in range(num_batches):
        message_bits = torch.randint(0, 2, (args.test_batch_size, args.block_len), dtype=torch.float).to(device)
        coded = turbo_encode(message_bits, trellis1, trellis2, interleaver, puncture = args.puncture).to(device)
        for k, snr in tqdm(enumerate(snr_range)):
            sigma = snr_db2sigma(snr)
            noise_variance = sigma**2

            if args.noise_type in ['awgn', 'fading', 't-dist', 'radar']:
                noisy_coded = corrupt_signal(coded, sigma, noise_type, vv = args.vv, radar_power = args.radar_power, radar_prob = args.radar_prob)
                received_llrs = 2*noisy_coded/noise_variance

            elif noise_type in ['EPA', 'EVA', 'ETU', 'MIMO']:
                message_bits = torch.from_numpy(msgs[:, ii, k]).to(device)
                received_llrs = torch.from_numpy(rx_llrs[:, ii, k]).to(device)

            if not only_tt:
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
            if args.input == 'y':
                tt_llrs, decoded_tt = tinyturbo_decode(weight_d, noisy_coded, trellis1, args.tinyturbo_iters, interleaver, method = args.tt_bcjr, puncture = args.puncture)
            else:
                tt_llrs, decoded_tt = tinyturbo_decode(weight_d, received_llrs, trellis1, args.tinyturbo_iters, interleaver, method = args.tt_bcjr, puncture = args.puncture)

            ber_tinyturbo = errors_ber(message_bits, decoded_tt[:, :-trellis1.total_memory])
            bler_tinyturbo = errors_bler(message_bits, decoded_tt[:, :-trellis1.total_memory])

            if ii == 0:
                bers_tt.append(ber_tinyturbo/num_batches)
                blers_tt.append(bler_tinyturbo/num_batches)
            else:
                bers_tt[k] += ber_tinyturbo/num_batches
                blers_tt[k] += bler_tinyturbo/num_batches

    return snr_range, bers_ml, bers_l, bers_tt, blers_ml, blers_l, blers_tt
