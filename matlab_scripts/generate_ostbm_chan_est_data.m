function [msg_data, enc_data, llr_data] = generate_ostbm_chan_est_data(num_tx, num_rx, msg_len, code_len, num_packets, EbNo, num_pilots, ch_known)
%OSTBC2M_E  Orthogonal space-time block coding with channel estimation for
%   2xM antenna configurations. 
%
%   BER2M_E = STBC2M_E(M, FRLEN, NUMPACKETS, EBNOVEC, PLEN) computes the
%   bit-error rate estimates via simulation for an orthogonal space-time
%   block coded configuration using two transmit antennas and M receive
%   antennas, where the frame length, number of packets simulated, Eb/No
%   range of values and the number of pilot symbols prepended per frame are
%   given by FRLEN, NUMPACKETS, EBNOVEC and PLEN parameters respectively.
%
%   The simulation uses the full-rate Alamouti encoding scheme for BPSK
%   modulated symbols with appropriate receiver combining. It uses the
%   pilot-aided Minimum-Mean-Square-Error (MMSE) method for estimating the
%   channel coefficients at the receiver. It is assumed the channel is
%   slowly fading (i.e. it remains constant for the whole frame of data and
%   changes independently from one frame to the other).
%
%   Suggested parameter values:
%       M = 1 or 2; FRLEN = 100; NUMPACKETS = 1000; EBNOVEC = 0:2:20, PLEN = 8;
%
%   Example:
%       ber22_e = ostbc2m_e(2, 100, 1000, 0:2:20, 8);
%
%   See also OSTBC2M, OSTBC4M, MRC1M.

%   References:
%   [1] A.F. Naguib, V. Tarokh, N. Seshadri, and A.R. Calderbank, "Space-time
%       codes for high data rate wireless communication: Mismatch analysis", 
%       Proceedings of IEEE International Conf. on Communications, 
%       pp. 309-313, June 1997.        
%
%   [2] S. M. Alamouti, "A simple transmit diversity technique for wireless 
%       communications", IEEE Journal on Selected Areas in Communications, 
%       Vol. 16, No. 8, Oct. 1998, pp. 1451-1458.

%   Copyright 2006-2017 The MathWorks, Inc.

%% Simulation parameters
rate = 1;   % Space-time block code rate

% Create comm.BPSKModulator and comm.BPSKDemodulator System objects
bpskMod = comm.BPSKModulator;
bpskDemod = comm.BPSKDemodulator('DecisionMethod','Log-likelihood ratio');

% Create comm.OSTBCEncoder and comm.OSTBCCombiner System objects
ostbcEnc = comm.OSTBCEncoder;
ostbcComb = comm.OSTBCCombiner( ...
    'NumReceiveAntennas', num_rx);

% Create a comm.MIMOChannel System object to simulate the 2xM spatially
% independent flat-fading Rayleigh channel
chanMIMO = comm.MIMOChannel( ...
    'MaximumDopplerShift', 0.001, ...
    'SpatialCorrelationSpecification', 'None', ...
    'NumTransmitAntennas', num_tx, ...
    'NumReceiveAntennas', num_rx, ...
    'PathGainsOutputPort', true);

% Create a comm.AWGNChannel System object. Set the NoiseMethod property of
% the channel to 'Signal to noise ratio (Eb/No)' to specify the noise level
% using the energy per bit to noise power spectral density ratio (Eb/No).
% The output of the BPSK modulator generates unit power signals; set the
% SignalPower property to 1 Watt.
chanAWGN = comm.AWGNChannel(...
    'NoiseMethod', 'Signal to noise ratio (Eb/No)',...
    'SignalPower', 1);

% Create a comm.ErrorRate calculator System object to evaluate BER.
errorCalc = comm.ErrorRate;

% Pilot sequences - orthogonal set over N
W = hadamard(num_pilots); % order gives the number of pilot symbols prepended/frame
pilots = W(:, 1:num_tx); 

%%  Pre-allocate variables for speed
HEst = zeros(code_len/rate, num_tx, num_rx); 

% Configure Turbo Enc/Dec
trellis = poly2trellis( 4, [ 13, 15 ], 13 );
intrlvrIndices =[0, 13,  6, 19, 12, 25, 18, 31, 24, 37, 30,  3, 36,  9,...
    2, 15,  8, 21, 14, 27, 20, 33, 26, 39, 32,  5, 38, 11,  4, 17,...
    10, 23, 16, 29, 22, 35, 28,  1, 34,  7] + 1;
turboenc = comm.TurboEncoder(trellis,intrlvrIndices);

msg_data = zeros(msg_len,num_packets,length(EbNo));
enc_data = zeros(code_len,num_packets,length(EbNo));
llr_data = zeros(code_len,num_packets,length(EbNo));

%% Loop over EbNo points
for idx = 1:length(EbNo)
    rng(idx,'twister');
    
    reset(errorCalc);
    chanAWGN.EbNo = EbNo(idx); 

    % Loop over the number of packets
    for packetIdx = 1:num_packets
        % Generate data vector per frame
        msg = randi([0 1], msg_len,1);
        msg_data(:,packetIdx,idx) = msg;

        % Encode using Turbo
        enc = turboenc(msg);
        enc_data(:,packetIdx,idx) = enc;

        % Modulate data
        modData = bpskMod(enc);

        % Alamouti Space-Time Block Encoder
        encData = ostbcEnc(modData);

        % Prepend pilot symbols for each frame
        txSig = [pilots; encData];

        % Pass through the 2xM channel
        reset(chanMIMO);
        [chanOut, H] = chanMIMO(txSig);

        % Add AWGN
        rxSig = chanAWGN(chanOut);
        
        % Channel Estimation
        %   For each link => N*M estimates
        HEst(1,:,:) = pilots(:,:).' * rxSig(1:num_pilots, :) / num_pilots;
        %   held constant for the whole frame
        HEst = HEst(ones(code_len/rate, 1), :, :);
        
        % Alamouti combiner
        if (ch_known == 1)
            rxDec = ostbcComb(rxSig(num_pilots+1:end,:), squeeze(H(num_pilots+1:end,:,:,:)));
        else
            rxDec = ostbcComb(rxSig(num_pilots+1:end,:), HEst);
        end

        % ML Detector (minimum Euclidean distance)
        llr_data(:,packetIdx,idx) = bpskDemod(rxDec); 
    end

end  % End of for loop for EbNo
    
