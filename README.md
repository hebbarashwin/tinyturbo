# tinyturbo

This repository includes:
- A PyTorch implementation of Turbo encoding and decoding.
- TINYTURBO - A neural Turbo decoder. Code for our paper "TinyTurbo: Efficient Turbo decoders on Edge"

All Python libraries required can be installed using:
```
pip install -r requirements.txt
```

## Training Neural Turbo decoder

Models are saved (with frequency specified by args.save_every) at Results/args.id/models/weights_{step_number}.pt

```
python main.py --batch_size 5000 --block_len 40 --target gt --loss_type BCE --init_type ones --num_steps 5000 --tinyturbo_iters 3 --turbo_iters 6 --train_snr -1 --lr 0.0008 --noise_type awgn --gpu 0 --id *string_of_your_choice* 
```

## Training Neural Turbo decoder from saved model checkpoint

```
python main.py --batch_size 1000 --block_len 40 --target gt --loss_type BCE --init_type ones --num_steps 5000 --tinyturbo_iters 3 --turbo_iters 6 --train_snr -1 --lr 0.0008 --noise_type awgn --gpu 0 --id *string_of_your_choice* --load_model_train *path to .pt file to initialize from*
```

## Testing Neural Turbo decoder

Tests the final model checkpoint at Results/args.id/models/weights.pt

```
python main.py --test_size 10000 --test_batch_size 10000 --block_len 40 --tinyturbo_iters 3 --turbo_iters 6 --noise_type awgn --gpu -1 --id *id of trained model* --test
```

## Testing Neural Turbo decoder at step_number

Tests the final model checkpoint at Results/args.id/models/weights_{step_number}.pt

```
python main.py --test_batch_size 10000 --block_len 40 --tinyturbo_iters 3 --turbo_iters 6 --noise_type awgn --gpu -1 --id *id of trained model* --test --load_model_step *step_number*
```

## Description of functions

- [convcode.py/conv_encode](https://github.com/hebbarashwin/tinyturbo/blob/5ca0d3050ec5747362a2b86cffeb47deddd8241d/convcode.py) : Convolutional code encoding
- [turbo.py/bcjr_decode](https://github.com/hebbarashwin/tinyturbo/blob/5ca0d3050ec5747362a2b86cffeb47deddd8241d/turbo.py) : BCJR (MAP) decoding of convolutional code
- [turbo.py/turbo_encode](https://github.com/hebbarashwin/tinyturbo/blob/5ca0d3050ec5747362a2b86cffeb47deddd8241d/turbo.py) : Turbo code encoding
- [turbo.py/turbo_decode](https://github.com/hebbarashwin/tinyturbo/blob/5ca0d3050ec5747362a2b86cffeb47deddd8241d/turbo.py) : Turbo decoder
- [turbonet.py/tinyturbo_decode](https://github.com/hebbarashwin/tinyturbo/blob/5ca0d3050ec5747362a2b86cffeb47deddd8241d/tinyturbo.py) : TINYTURBO decoder
- [turbonet.py/train](https://github.com/hebbarashwin/tinyturbo/blob/5ca0d3050ec5747362a2b86cffeb47deddd8241d/tinyturbo.py) : Train TINYTURBO
- [turbonet.py/test](https://github.com/hebbarashwin/tinyturbo/blob/5ca0d3050ec5747362a2b86cffeb47deddd8241d/tinyturbo.py) : Test TINYTURBO
