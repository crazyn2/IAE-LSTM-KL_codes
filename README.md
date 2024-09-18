# PyTorch Implementation of Deep SVDD
This repository provides a [PyTorch](https://pytorch.org/) implementation of the *IAE-LSTM-KL* method presented in our
IEEE TIM 2024 paper "mproved AutoEncoder with LSTM Module and KL Divergence for Anomaly Detection".


## Citation and Contact
You find a PDF of the Deep One-Class Classification ICML 2018 paper at 
[https://ieeexplore.ieee.org/document/10680570](https://ieeexplore.ieee.org/document/10680570).

If you use our work, please also cite the paper:
```
@ARTICLE{10680570,

  author={Huang, Wei and Zhang, Bingyang and Zhang, Kaituo and Gao, Hua and Wan, Rongchun},

  journal={IEEE Transactions on Instrumentation and Measurement}, 

  title={Improved AutoEncoder with LSTM Module and KL Divergence for Anomaly Detection}, 

  year={2024},

  volume={},

  number={},

  pages={1-1},

  keywords={Data models;Long short term memory;Vectors;Anomaly detection;Feature extraction;Training;Mathematical models;LSTM;Deep SVDD;autoencoder;hypersphere collapse;anomaly detection},

  doi={10.1109/TIM.2024.3460931}
}
```

If you would like to get in touch, please contact [aschangeme@outlook.com](mailto:aschangeme@outlook.com).


## Installation
This code is written in `Python 3.10` and requires the packages listed in `requirements.txt`.

Clone the repository to your local machine and directory of choice:
```
git clone https://github.com/lukasruff/Deep-SVDD-PyTorch.git
```

To run the code, we recommend setting up a virtual environment, e.g. using `virtualenv` or `conda`:

### `conda`
```
cd <path>
conda create --name myenv
source activate myenv
while read requirement; do conda install -n myenv --yes $requirement; done < requirements.txt
```


## Running experiments

We currently have implemented the FMNIST, CIFAR-10, MvTec and WTBI datasets and 
simple LeNet-type networks.

## License
MIT