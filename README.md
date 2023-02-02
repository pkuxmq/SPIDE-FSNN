# SPIDE-FSNN
This is the PyTorch implementation of paper "SPIDE: A Purely Spike-based Method for Training Feedback Spiking Neural Networks" **(Neural Networks, 2023)**. [[link]](https://doi.org/10.1016/j.neunet.2023.01.026) [[arxiv]](https://arxiv.org/abs/2302.00232)

## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch, torchvision](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python`

## Training
Run as following with some hyperparameters:

	python mnist_lenet.py --path path_to_data_dir --time_step 30 --time_step_back 100 --scale_factor 100 -c checkpoint_name --gpu-id 0

	python cifar_alexnetf.py --dataset cifar10 --path path_to_data_dir --time_step 30 --time_step_back 250 --scale_factor 400 -c checkpoint_name --gpu-id 0

	python cifar_cifarnetf.py --dataset cifar100 --path path_to_data_dir --time_step 30 --time_step_back 250 --scale_factor 500 -c checkpoint_name --gpu-id 0

	python dvs_cifar_convnetf.py --path path_to_data_dir --time_step 30 --time_step_back 250 --scale_factor 400 -c checkpoint_name --gpu-id 0

The leaky term is set as 1 for the IF model, and should be set in the range of (0, 1) for the LIF model. The default hyperparameters in the code are mostly the same as in the paper (except that the scale factor is 400 for CIFAR-10 and 500 for CIFAR-100).

## Testing
Run as following with some hyperparameters:

	python cifar_alexnetf.py --dataset cifar10 --path path_to_data_dir --time_step 30 -c checkpoint_name --resume path_to_checkpoint --gpu-id 0 --evaluate

## Acknowledgement

The codes are modified from the [IDE-FSNN](https://github.com/pkuxmq/IDE-FSNN) repository. Some codes for the data prepoccessing of CIFAR10-DVS are from the [spikingjelly](https://github.com/fangwei123456/spikingjelly) repository, and the codes for some utils are from the [pytorch-classification](https://github.com/bearpaw/pytorch-classification) repository.

## Contact
If you have any questions, please contact <mingqing_xiao@pku.edu.cn>.
