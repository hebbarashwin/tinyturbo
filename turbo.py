__author__ = 'hebbarashwin'

import numpy as np
import torch

from convcode import conv_encode
from interleaver import Interleaver
from utils import dec2bitarray


def max_star(metrics):

    assert metrics.shape[1] >= 2, "Number of operants for max* operation must be at least 2"
    temp = metrics[:, 0].clone()
    for ii in range(1, metrics.shape[1]):
        temp = torch.max(temp, metrics[:, ii].clone()) + torch.log(1 + torch.exp(-torch.abs(temp - metrics[:, ii].clone())))

    return temp


def turbo_encode(message_bits, trellis1, trellis2, interleaver, puncture=False):
    """ Turbo Encoder.
    Encode Bits using a parallel concatenated rate-1/3
    turbo code consisting of two rate-1/2 systematic
    convolutional component codes.
    Parameters
    ----------
    message_bits : 2D torch Tensor containing {0, 1} of shape (batch_size, M)
        Stream of bits to be turbo encoded.
    trellis1 : Trellis object
        Trellis representation of the
        first code in the parallel concatenation.
    trellis2 : Trellis object
        Trellis representation of the
        second code in the parallel concatenation.
    interleaver : Interleaver object
        Interleaver used in the turbo code.
    puncture: Bool
        Currently supports only puncturing pattern '110101'
    Returns
    -------
    stream : torch Tensor of turbo encoded codewords, of shape (batch_size, 3*M + 4*memory)
            where memory is the number of delay elements in the convolutional code, M is the message length.

            First 3*M bits are [sys_1, non_sys1_1, non_sys2_1, . . . . sys_j, non_sys1_j, non_sys2_j, . . . sys_M, non_sys1_M, non_sys2_M]
            Next 2*memory bits are termination bits of sys and non_sys1 : [sys_term_1, non_sys1_term_1, . . . . sys_term_j, non_sys1_term_j, . . . sys_term_M, non_sys1_term_M]
            Next 2*memory bits are termination bits of sys_interleaved and non_sys2 : [sys_inter_term_1, non_sys2_term_1, . . . . sys_inter_term_j, non_sys2_term_j, . . . sys_inter_term_M, non_sys2_term_M]

        Encoded bit streams corresponding
        to the systematic output
        and the two non-systematic
        outputs from the two component codes.
    """

    block_len = message_bits.shape[1]
    stream = conv_encode(message_bits, trellis1)
    sys_stream = stream[:, ::2]
    non_sys_stream_1 = stream[:, 1::2]

    interlv_msg_bits = interleaver.interleave(message_bits)
    #puncture_matrix = np.array([[0, 1]])
    stream_int = conv_encode(interlv_msg_bits, trellis2)
    sys_stream_int = stream_int[:, ::2]
    non_sys_stream_2 = stream_int[:, 1::2]

    #Termination bits
    term_sys1 = sys_stream[:, -trellis1.total_memory:]
    term_sys2 = sys_stream_int[:, -trellis2.total_memory:]
    term_nonsys1 = non_sys_stream_1[:, -trellis1.total_memory:]
    term_nonsys2 = non_sys_stream_2[:, -trellis2.total_memory:]

    sys_stream = sys_stream[:, :-trellis1.total_memory]
    non_sys_stream_1 = non_sys_stream_1[:, :-trellis1.total_memory]
    non_sys_stream_2 = non_sys_stream_2[:, :-trellis2.total_memory]

    codeword = torch.empty((message_bits.shape[0], message_bits.shape[1]*3), dtype=sys_stream.dtype)
    codeword[:, 0::3] = sys_stream
    codeword[:, 1::3] = non_sys_stream_1
    codeword[:, 2::3] = non_sys_stream_2
    term1 = stream[:, -2*trellis1.total_memory:]
    term2 = stream_int[:, -2*trellis1.total_memory:]

    if not puncture:
        out = torch.cat((codeword, term1, term2), dim=1)
    else:
        inds = torch.Tensor([1, 1, 0, 1, 0, 1]).repeat(block_len//2).byte()
        punctured_codeword = codeword[:, inds]
        out = torch.cat((punctured_codeword, term1, term2), dim=1)
    return out

def bcjr_decode(sys_llrs, non_sys_llrs, trellis, L_int, method = 'max_log_MAP'):
    """ BCJR Decoder.
    Decode a rate-1/2 systematic convoluitonal code.

    Parameters
    ----------
    sys_llrs : systematic LLRs of shape (batch_size, 3*M + 4*memory)
        Received LLRs corresponding to systematic bits
    non_sys_llrs : non-systematic LLRs of shape (batch_size, 3*M + 4*memory)
        Received LLRs corresponding to non-systematic parity bits
    trellis : Trellis object
        Trellis representation of the convolutional code
    L_int : intrinsic LLRs of shape (batch_size, 3*M + 4*memory)
        Intrinsic LLRs (prior). (Set to zeros if no prior)
    method : Turbo decoding method
        max-log-MAP or MAP

    Returns
    -------
    L_ext : torch Tensor of decoded LLRs, of shape (batch_size, M + memory)

    decoded_bits: L_ext > 0

        Decoded beliefs
    """

    if method not in ['max_log_MAP', 'MAP']:
        method = 'MAP'

    k = trellis.k
    n = trellis.n
    rate = float(k)/n
    number_states = trellis.number_states
    number_inputs = trellis.number_inputs

    batch_size, msg_length = sys_llrs.shape

    next_state_table = trellis.next_state_table
    output_table = trellis.output_table
    # Initialize forward state metrics (alpha)
    f_state_metrics = -1000* torch.ones((batch_size, number_states, msg_length+1)).to(sys_llrs.device)
    f_state_metrics[:, 0, 0] = 0
    f_state_temp = torch.zeros((batch_size, number_states, number_inputs)).to(sys_llrs.device)

    # Initialize backward state metrics (beta)
    b_state_metrics = -1000* torch.ones((batch_size, number_states, msg_length+1)).to(sys_llrs.device)
    # b_state_metrics[:, :,msg_length] = 0
    b_state_metrics[:, 0,msg_length] = 0
    b_state_temp = torch.zeros((batch_size, number_inputs)).to(sys_llrs.device)

    branch_probs = torch.zeros((batch_size, number_inputs, number_states, msg_length+1)).to(sys_llrs.device)
    L_ext = torch.zeros_like(sys_llrs)
    L_temp = [[], []]
    # Backward recursion:
    for reverse_time_index in range(msg_length, 0, -1):
        for current_state in range(number_states):
            for current_input in range(number_inputs):
                next_state = next_state_table[current_state, current_input]
                code_symbol = output_table[current_state, current_input]
                codeword_array = dec2bitarray(code_symbol, n)
                parity_bit = codeword_array[1]
                msg_bit = codeword_array[0]

                code_symbol_0 = 2*codeword_array[0]-1
                code_symbol_1 = 2*codeword_array[1]-1

                rx_llr_0 = sys_llrs[:, reverse_time_index-1]
                rx_llr_1 = non_sys_llrs[:, reverse_time_index-1]

                # log of branch prob :
                # branch_prob = -(x**2 + y**2)/(2*noise_variance) + torch.log(torch.sigmoid((1 - 2*current_input)*L_int))
                # branch_prob = (code_symbol_0*rx_symbol_0 + code_symbol_1*rx_symbol_1)/noise_variance + 0.5*(2*current_input-1)*L_int[:, reverse_time_index-1]

                branch_prob = 0.5*(code_symbol_0*rx_llr_0 + code_symbol_1*rx_llr_1) + 0.5*(2*current_input-1)*L_int[:, reverse_time_index-1]
                branch_probs[:, current_input, current_state, reverse_time_index-1] = branch_prob

                b_state_temp[:, current_input] = b_state_metrics[:, next_state, reverse_time_index] + branch_prob
            if method == 'max_log_MAP':
                b_state_metrics[:, current_state, reverse_time_index-1] , _ = torch.max(b_state_temp, dim=1)
            elif method == 'MAP':
                b_state_metrics[:, current_state, reverse_time_index-1] = max_star(b_state_temp)

    # Forward recursion:
    for time_index in range(1, msg_length+1):
        L_temp = [[], []]
        for current_state in range(number_states):
            for current_input in range(number_inputs):
                next_state = next_state_table[current_state, current_input]
                branch_prob = branch_probs[:, current_input, current_state, time_index-1]
                f_state_temp[:, next_state, current_input] = f_state_metrics[:, current_state, time_index-1] + branch_prob
        if method == 'max_log_MAP':
            for s in range(number_states):
                f_state_metrics[:, s, time_index] , _ = torch.max(f_state_temp[:, s, :], dim=1)
        elif method == 'MAP':
            for s in range(number_states):
                f_state_metrics[:, s, time_index] = max_star(f_state_temp[:, s, :])
        for current_state in range(number_states):
            for current_input in range(number_inputs):
                next_state = next_state_table[current_state, current_input]
                branch_prob = branch_probs[:, current_input, current_state, time_index-1]
                alpha = f_state_metrics[:, current_state, time_index-1]
                beta = b_state_metrics[:, next_state, time_index]
                L = alpha + beta + branch_prob

                L_temp[current_input].append(L)
        stack0 = torch.stack(L_temp[0], -1)
        stack1 = torch.stack(L_temp[1], -1)

        if method == 'max_log_MAP':
            L0, _ = torch.max(stack0, -1)
            L1, _ = torch.max(stack1, -1)
            L_ext[:, time_index-1] = L1 - L0
        elif method == 'MAP':
            L0 = max_star(stack0)
            L1 = max_star(stack1)
            L_ext[:, time_index-1] = L1 - L0

    decoded = (L_ext> 0).float()
    return L_ext, decoded

def turbo_decode(received_llrs, trellis, number_iterations, interleaver, L_int = None, method = 'max_log_MAP', puncture=False):

    """ Turbo Decoder.
    Decode a Turbo code.

    Parameters
    ----------
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


    # puncturing to get rate 1/2 . Pattern: '110101'. Can change this later for more patterns
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
        L_int = torch.zeros_like(sys_llrs)

    L_int_1 = L_int

    for iteration in range(number_iterations):
        # [L_ext_1, decoded] = bcjr_decode(sys_llrs, non_sys_llrs1, trellis, L_int_1, method=method)
        #
        # L_ext_1 = L_ext_1 - L_int_1 - sys_llr
        # L_int_2 = interleaver.interleave(L_ext_1[:, :sys_stream.shape[1]])
        # L_int_2 = torch.cat((L_int_2, torch.zeros_like(term_sys1)), -1)
        #
        # [L_ext_2, decoded] = bcjr_decode(sys_llrs_inter, non_sys_llrs2, trellis, L_int_2, method=method)
        #
        # L_ext_2 = L_ext_2 - L_int_2
        # L_int_1 = interleaver.deinterleave(L_ext_2[:, :sys_stream.shape[1]])
        # L_int_1 = L_int_1 - sys_llr[:, :sys_stream.shape[1]]
        # L_int_1 = torch.cat((L_int_1, torch.zeros_like(term_sys1)), -1)

        [L_ext_1, decoded] = bcjr_decode(sys_llrs, non_sys_llrs1, trellis, L_int_1, method=method)

        L_ext = L_ext_1 - L_int_1 - sys_llr
        L_e_1 = L_ext_1[:, :sys_stream.shape[1]]
        L_1 = L_int_1[:, :sys_stream.shape[1]]

        L_int_2 = L_e_1 - sys_llr[:, :sys_stream.shape[1]] - L_1
        L_int_2 = interleaver.interleave(L_int_2)
        L_int_2 = torch.cat((L_int_2, torch.zeros_like(term_sys1)), -1)

        [L_ext_2, decoded] = bcjr_decode(sys_llrs_inter, non_sys_llrs2, trellis, L_int_2, method=method)

        L_e_2 = interleaver.deinterleave(L_ext_2[:, :sys_stream.shape[1]])
        L_2 = interleaver.deinterleave(L_int_2[:, :sys_stream.shape[1]])

        L_int_1 = L_e_2 - sys_llr[:, :sys_stream.shape[1]] - L_2
        L_int_1 = torch.cat((L_int_1, torch.zeros_like(term_sys1)), -1)
    LLRs = L_ext + L_int_1 + sys_llr
    decoded_bits = (LLRs > 0).float()

    return LLRs, decoded_bits
