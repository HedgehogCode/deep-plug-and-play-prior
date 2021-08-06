# Deep Plug-and-Play Priors for Image Reconstruction

This repository implements two algorithms for image reconstruction using a Gaussian denoiser in TensorFlow 2.

Algorithms:

- **DMSP:** S. A. Bigdeli, P. Favaro, M. Jin, and M. Zwicker, [“Deep Mean-Shift Priors for Image Restoration,”](http://papers.nips.cc/paper/6678-deep-mean-shift-priors-for-image-restoration.pdf) in Advances in Neural Information Processing Systems 30, 2017, pp. 763–772.
- **HQS:** K. Zhang, Y. Li, W. Zuo, L. Zhang, L. Van Gool, and R. Timofte, [“Plug-and-Play Image Restoration with Deep Denoiser Prior,”](http://arxiv.org/abs/2008.13751) IEEE Trans. Pattern Anal. Mach. Intell., pp. 1–1, 2021

## Getting Started

Install using pip

```
$ pip install git+https://github.com/HedgehogCode/deep-plug-and-play-prior
```

## Usage

See the example [notebooks](notebooks/) for usage examples.

## Related Repositories

DMSP Implementations (significantly slower):

- Matlab: https://github.com/siavashBigdeli/DMSP (only deblur)
- TensorFlow 1: https://github.com/siavashBigdeli/DMSP-tensorflow (only deblur)
- TensorFlow 2: https://github.com/siavashBigdeli/DMSP-TF2 (deblur and sr (but not bicubic))

HQS:

- Matlab: https://github.com/cszn/IRCNN
- Pytorch: https://github.com/cszn/DPIR
