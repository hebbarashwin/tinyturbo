% STBC 2x2 with channel est
num_tx = 2;
num_rx = 2;
msg_len = 40;
code_len = 132;
EbNo = -5:1:5;
num_packets = 10;
num_pilots = 2; % min. 2 pilots per frame; here frame 1 is 1 codeword
ch_known = 0; % change to 1 for perfect CSI
[msg_data, enc_data, llr_data] = generate_ostbm_chan_est_data(num_tx, num_rx, msg_len, code_len, num_packets, EbNo, num_pilots, ch_known);

%% sanity check by decoding using raw LLRs
enc_data_est = llr_data < 0;
ber_raw = squeeze(mean(enc_data ~= enc_data_est,[1 2]));
disp("Raw ber :")
disp(ber_raw');