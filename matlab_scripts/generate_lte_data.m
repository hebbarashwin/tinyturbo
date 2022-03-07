function [msg_data, enc_data, llr_data] = generate_lte_data(coding_scheme, msg_len, code_len, chan, SNRs, num_blocks, num_sym)

% load default code params
if (coding_scheme == "BCH")
    load('G_BCH_63_36.mat','G');
    G = double(G);
else
    trellis = poly2trellis( 4, [ 13, 15 ], 13 );
    intrlvrIndices = [0, 13,  6, 19, 12, 25, 18, 31, 24, 37, 30,  3, 36,  9,...
                        2, 15,  8, 21, 14, 27, 20, 33, 26, 39, 32,  5, 38, 11,  4, 17,...
                        10, 23, 16, 29, 22, 35, 28,  1, 34,  7] + 1;
    turboenc = comm.TurboEncoder(trellis,intrlvrIndices);
end

msg_data = randi([0 1],msg_len,num_sym*num_blocks,length(SNRs));
enc_data = zeros(code_len,num_sym*num_blocks,length(SNRs));
rx_data = zeros(code_len,num_sym*num_blocks,length(SNRs));

for i_SNR = 1:length(SNRs)
    cur_SNR = SNRs(i_SNR);

    for i_sym = 1:num_sym
        start_ind = (i_sym - 1)*num_blocks + 1;
        end_ind = i_sym*num_blocks;
        
        msg = squeeze(msg_data(:,start_ind:end_ind, i_SNR));
        enc = zeros(code_len,num_blocks); %179 by default
        % encode the msg data
        if (coding_scheme == "BCH")
            enc = mod(G*msg,2);
        else
            for i = 1:num_blocks
                enc(:,i) = turboenc(msg_data(:,i,i_SNR));
            end
        end
        enc_data(:,start_ind:end_ind, i_SNR) = enc;
        inputBits = [enc(:);randi([0 1],26028 - num_blocks*code_len,1)]; %padding
        rx_val = lte_channel(inputBits, cur_SNR, chan);
        rx_val = rx_val(1:num_blocks*code_len); % remove padding
        rx_data(:,start_ind:end_ind, i_SNR) = reshape(rx_val,code_len,num_blocks);
    end

end

% demod using matlab bpsk demod
bpskDemod = comm.BPSKDemodulator('DecisionMethod','Log-likelihood ratio');
llr_data = reshape(bpskDemod(rx_data(:)),size(rx_data));

end


function detLLR = lte_channel(inputBits, SNRdB, c_type)


%% Cell-Wide Settings

enb.NDLRB = 15;                 % Number of resource blocks
enb.CellRefP = 1;               % One transmit antenna port
enb.NCellID = 10;               % Cell ID
enb.CyclicPrefix = 'Normal';    % Normal cyclic prefix
enb.DuplexMode = 'FDD';         % FDD

%% SNR Configuration
SNR = 10^(SNRdB/20);    % Linear SNR
rng('default');         % Configure random number generators


%% Channel Model Configuration
cfg.Seed = ceil(100*rand());                  % Channel seed
cfg.NRxAnts = 1;               % 1 receive antenna
cfg.DelayProfile = c_type;     % EVA delay spread
cfg.DopplerFreq = 0;           % 120Hz Doppler frequency
cfg.MIMOCorrelation = 'Low';   % Low (no) MIMO correlation
cfg.InitTime = 0;              % Initialize at time zero
cfg.NTerms = 16;               % Oscillators used in fading model
cfg.ModelType = 'GMEDS';       % Rayleigh fading model type
cfg.InitPhase = 'Random';      % Random initial phases
cfg.NormalizePathGains = 'On'; % Normalize delay profile power
cfg.NormalizeTxAnts = 'On';    % Normalize for transmit antennas

%% Channel Estimator Configuration

cec.PilotAverage = 'UserDefined'; % Pilot averaging method
cec.FreqWindow = 9;               % Frequency averaging window in REs
cec.TimeWindow = 9;               % Time averaging window in REs

cec.InterpType = 'Cubic';         % Cubic interpolation
cec.InterpWinSize = 3;            % Interpolate up to 3 subframes
% simultaneously
cec.InterpWindow = 'Centred';     % Interpolation windowing method

%% Subframe Resource Grid Size

gridsize = lteDLResourceGridSize(enb);
K = gridsize(1);    % Number of subcarriers
L = gridsize(2);    % Number of OFDM symbols in one subframe
P = gridsize(3);    % Number of transmit antenna ports

%% Transmit Resource Grid

txGrid = [];

%% Payload Data Generation

% Number of bits needed is size of resource grid (K*L*P) * number of bits
% per symbol (2 for QPSK)
% % numberOfBits = K*L*P*2;

% Modulate input bits
inputSym = lteSymbolModulate(inputBits,'BPSK');

%% Frame Generation
X_new = 0;
% For all subframes within the frame
for sf = 0:10
    
    % Set subframe number
    enb.NSubframe = mod(sf,10);
    
    % Generate empty subframe
    subframe = lteDLResourceGrid(enb);
    
    % Generate synchronizing signals
    pssSym = ltePSS(enb);
    sssSym = lteSSS(enb);
    pssInd = ltePSSIndices(enb);
    sssInd = lteSSSIndices(enb);
    
    % Map synchronizing signals to the grid
    subframe(pssInd) = pssSym;
    subframe(sssInd) = sssSym;
    
    % Generate cell specific reference signal symbols and indices
    cellRsSym = lteCellRS(enb);
    cellRsInd = lteCellRSIndices(enb);
    X_old = X_new;
    X_new = X_new+180*14-(length(cellRsInd)+length(sssInd)+length(pssInd));
    % Map cell specific reference signal to grid
    subframe(cellRsInd) = cellRsSym;
    
    % Map input symbols to grid
    dataIdx = [cellRsInd;pssInd;sssInd];
    subframe(setdiff(1:180*14,dataIdx)) = inputSym(X_old+1:X_new);
    
    % Append subframe to grid to be transmitted
    txGrid = [txGrid subframe]; %#ok
    
end

%% OFDM Modulation

[txWaveform,info] = lteOFDMModulate(enb,txGrid);

%% Fading Channel
cfg.SamplingRate = info.SamplingRate;

% Pass data through the fading channel model
rxWaveform = lteFadingChannel(cfg,txWaveform);

%% Additive Noise

% Calculate noise gain
N0 = 1/(sqrt(2.0*enb.CellRefP*double(info.Nfft))*SNR);

% Create additive white Gaussian noise
noise = N0*complex(randn(size(rxWaveform)),randn(size(rxWaveform)));

% Add noise to the received time domain waveform
rxWaveform = rxWaveform + noise;


%% Synchronization
offset = lteDLFrameOffset(enb,rxWaveform);
rxWaveform = rxWaveform(1+offset:end,:);

%% OFDM Demodulation

rxGrid = lteOFDMDemodulate(enb,rxWaveform);

%% Channel Estimation

enb.NSubframe = 0;
[estChannel, noiseEst] = lteDLChannelEstimate(enb,cec,rxGrid);

%% MMSE Equalization

eqGrid = lteEqualizeMMSE(rxGrid, estChannel, noiseEst);

%% Frame Extraction
detLLR = zeros(23752,1);
X_new = 0;
% For all subframes within the frame
for sf = 0:9
    
    % Set subframe number
    enb.NSubframe = mod(sf,10);
    
    % Generate synchronizing signals indices
    pssInd = ltePSSIndices(enb);
    sssInd = lteSSSIndices(enb);
    
    % Generate cell specific reference signal indices
    cellRsInd = lteCellRSIndices(enb);
    
    X_old = X_new;
    X_new = X_new+180*14-(length(cellRsInd)+length(sssInd)+length(pssInd));
    
    % Map input symbols to grid
    dataIdx = [cellRsInd;pssInd;sssInd];
    RxSig = eqGrid(:,sf*14 + 1: (sf + 1)*14);
    detLLR(X_old+1:X_new) = RxSig(setdiff(1:180*14,dataIdx));
end

detLLR = detLLR(:);

end
