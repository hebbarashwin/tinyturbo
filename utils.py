import numpy as np
import torch
import torch.nn.functional as F

def dec2bitarray(in_number, bit_width):
    """
    Converts a positive integer to NumPy array of the specified size containing
    bits (0 and 1).

    Parameters
    ----------
    in_number : int
        Positive integer to be converted to a bit array.

    bit_width : int
        Size of the output bit array.

    Returns
    -------
    bitarray : 1D ndarray of ints
        Array containing the binary representation of the input decimal.

    """

    binary_string = bin(in_number)
    length = len(binary_string)
    bitarray = np.zeros(bit_width, 'int')
    for i in range(length-2):
        bitarray[bit_width-i-1] = int(binary_string[length-i-1])

    return bitarray

def bitarray2dec(in_bitarray):
    """
    Converts an input NumPy array of bits (0 and 1) to a decimal integer.

    Parameters
    ----------
    in_bitarray : 1D ndarray of ints
        Input NumPy array of bits.

    Returns
    -------
    number : int
        Integer representation of input bit array.
    """

    number = 0

    for i in range(len(in_bitarray)):
        number = number + in_bitarray[i]*pow(2, len(in_bitarray)-1-i)

    return number

def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)

def errors_ber(y_true, y_pred):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred)).float()
    res = sum(sum(myOtherTensor))/(myOtherTensor.shape[0]*myOtherTensor.shape[1])
    return res.item()


def errors_bler(y_true, y_pred):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    decoded_bits = torch.round(y_pred).cpu()
    X_test       = torch.round(y_true).cpu()
    tp0 = (abs(decoded_bits-X_test)).view([X_test.shape[0],X_test.shape[1]])
    tp0 = tp0.detach().cpu().numpy()
    bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])
    return bler_err_rate

def corrupt_signal(input_signal, sigma = 1.0, noise_type = 'awgn', vv =5.0, radar_power = 20.0, radar_prob = 5e-2):

    data_shape = input_signal.shape  # input_signal has to be a numpy array.
    assert noise_type in ['awgn', 'fading', 'radar', 't-dist'], "Invalid noise type"

    if noise_type == 'awgn':
        noise = sigma * torch.randn_like(input_signal) # Define noise
        corrupted_signal = 2.0*input_signal-1.0 + noise

    elif noise_type == 'fading':
        fading_h = torch.sqrt(torch.randn_like(input_signal)**2 +  torch.randn_like(input_signal)**2)/np.sqrt(3.14/2.0)
        noise = sigma * torch.randn_like(input_signal) # Define noise
        corrupted_signal = fading_h *(2.0*input_signal-1.0) + noise

    elif noise_type == 'radar':
        add_pos     = np.random.choice([0.0, 1.0], data_shape,
                                       p=[1 - radar_prob, radar_prob])

        corrupted_signal = radar_power* np.random.standard_normal( size = data_shape ) * add_pos
        noise = sigma * torch.randn_like(input_signal) +\
                    torch.from_numpy(corrupted_signal).type(torch.FloatTensor).to(input_signal.device)
        corrupted_signal = 2.0*input_signal-1.0 + noise

    elif noise_type == 't-dist':
        noise = sigma * np.sqrt((vv-2)/vv) *np.random.standard_t(vv, size = data_shape)
        corrupted_signal = 2.0*input_signal-1.0 + torch.from_numpy(noise).type(torch.FloatTensor).to(input_signal.device)

    return corrupted_signal

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
