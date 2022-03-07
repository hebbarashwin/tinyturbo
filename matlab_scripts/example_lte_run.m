coding_scheme= "Turbo";
if (coding_scheme == "BCH")
    msg_len = 36;
    code_len = 63;
else
    msg_len = 40;
    code_len = 132;
end
n_blocks = 179;
chan = 'ETU'; % EPA/EVA/ETU
SNRs = 1:1:10;
num_blocks = 179; % do not change unless needed
num_sym = 10; % increase to generate more data

[msg_data, enc_data, llr_data] = generate_lte_data(coding_scheme, msg_len, code_len, chan, SNRs, num_blocks, num_sym);

disp("Finished generating data");

%% sanity check by decoding using raw LLRs
enc_data_est = llr_data < 0;
ber_raw = squeeze(mean(enc_data ~= enc_data_est,[1 2]));
disp("Raw ber :")
disp(ber_raw');
