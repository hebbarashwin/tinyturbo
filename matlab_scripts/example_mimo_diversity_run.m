% Rayleigh distributed MIMO channel
% Choose 1x1 for baseline, 2x1 for Alamouti Tx diversity, 1x2 for MRC Rx
% diversity
% Added only Turbo (40,132) support for now

num_tx = 1;
num_rx = 2;
max_num_tx = 2; % needed for internal comp
max_num_rx = 2; % needed for internal comp
msg_len = 40;
code_len = 132;
EbNo = -5:1:5;
num_packets = 10;

% final data will be of size : (msg_len/code_len, num_packets, length(EbNo) 
[msg_data, enc_data, llr_data] = generate_mimo_diversity_data (num_tx, num_rx, max_num_tx, max_num_rx, msg_len, code_len, EbNo, num_packets);

%% sanity check by decoding using raw LLRs
enc_data_est = llr_data < 0;
ber_raw = squeeze(mean(enc_data ~= enc_data_est,[1 2]));
disp("Raw ber :")
disp(ber_raw');
