function [msg_data, enc_data, llr_data] = generate_mimo_diversity_data (num_tx, num_rx, max_num_tx, max_num_rx, msg_len, code_len, EbNo, num_packets)
  
    % Create comm.BPSKModulator and comm.BPSKDemodulator System objects(TM)
    bpskMod = comm.BPSKModulator;
    bpskDemod = comm.BPSKDemodulator('DecisionMethod','Log-likelihood ratio');
    
    % Create comm.OSTBCEncoder and comm.OSTBCCombiner System objects
    ostbcEnc = comm.OSTBCEncoder;
    ostbcComb = comm.OSTBCCombiner;
    
    % Create two comm.AWGNChannel System objects for one and two receive
    % antennas respectively. Set the NoiseMethod property of the channel to
    % 'Signal to noise ratio (Eb/No)' to specify the noise level using the
    % energy per bit to noise power spectral density ratio (Eb/No). The output
    % of the BPSK modulator generates unit power signals; set the SignalPower
    % property to 1 Watt.
    awgn1Rx = comm.AWGNChannel(...
        'NoiseMethod', 'Signal to noise ratio (Eb/No)', ...
        'SignalPower', 1);
    awgn2Rx = clone(awgn1Rx);
    
    % Since the comm.AWGNChannel System objects as well as the RANDI function
    % use the default random stream, the following commands are executed so
    % that the results will be repeatable, i.e., same results will be obtained
    % for every run of the example. The default stream will be restored at the
    % end of the example.
    s = rng(55408);
    
    % Pre-allocate variables for speed
    H = zeros(code_len, max_num_tx, max_num_rx);
    
    % Configure Turbo Enc/Dec
    trellis = poly2trellis( 4, [ 13, 15 ], 13 );
    intrlvrIndices =[0, 13,  6, 19, 12, 25, 18, 31, 24, 37, 30,  3, 36,  9,...
        2, 15,  8, 21, 14, 27, 20, 33, 26, 39, 32,  5, 38, 11,  4, 17,...
        10, 23, 16, 29, 22, 35, 28,  1, 34,  7] + 1;
    turboenc = comm.TurboEncoder(trellis,intrlvrIndices);
  
    msg_data = zeros(msg_len,num_packets,length(EbNo));
    enc_data = zeros(code_len,num_packets,length(EbNo));
    llr_data = zeros(code_len,num_packets,length(EbNo));

    % Loop over several EbNo points
    for idx = 1:length(EbNo)
        % Set the EbNo property of the AWGNChannel System objects
        awgn1Rx.EbNo = EbNo(idx);
        awgn2Rx.EbNo = EbNo(idx);

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
    
            % Create the Rayleigh distributed channel response matrix
            %   for two transmit and two receive antennas
            H(1:max_num_tx:end, :, :) = (randn(code_len/2, max_num_tx, max_num_rx) + ...
                             1i*randn(code_len/2, max_num_tx, max_num_rx))/sqrt(2);
            %   assume held constant for 2 symbol periods
            H(2:max_num_tx:end, :, :) = H(1:max_num_tx:end, :, :);
    
            % Extract part of H to represent the 1x1, 2x1 and 1x2 channels
            H11 = H(:,1,1);
            H21 = H(:,:,1)/sqrt(2);
            H12 = squeeze(H(:,1,:));
    
            % Pass through the channels
            chanOut11 = H11 .* modData;
            chanOut21 = sum(H21.* encData, 2);
            chanOut12 = H12 .* repmat(modData, 1, 2);
    
            % Add AWGN
            rxSig11 = awgn1Rx(chanOut11);
            rxSig21 = awgn1Rx(chanOut21);
            rxSig12 = awgn2Rx(chanOut12);
            
    
            % Alamouti Space-Time Block Combiner
            decData = ostbcComb(rxSig21, H21);
    
            % ML Detector (minimum Euclidean distance)
            if (num_tx == 1 && num_rx == 1)
                llr = bpskDemod(rxSig11.*conj(H11));
            elseif (num_tx == 2 && num_rx == 1)
                llr = bpskDemod(decData);
            elseif (num_tx == 1 && num_rx == 2)
                llr = bpskDemod(sum(rxSig12.*conj(H12), 2));
            end
            
            % store for block processing
            llr_data(:,packetIdx,idx) = llr;
    
        end
     
    end  % end of for loop for EbNo
    
    rng(s);

end
