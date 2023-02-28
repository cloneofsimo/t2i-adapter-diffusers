# T2I Adapter Diffusers

Developer-friendly port of the [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter), [Paper](https://arxiv.org/abs/2302.08453) to `diffusers`.

## Installation

```bash
pip install git+https://github.com/cloneofsimo/t2i-adapter-diffusers
```

## Usage

Example code Using all of the adapters is at `test_all.py`. In short, you need to substitute `UNet2DConditionModel` to `T2IAdapterUNet2DConditionModel`.

## References

```bibtex
@misc{https://doi.org/10.48550/arxiv.2302.08453,
  doi = {10.48550/ARXIV.2302.08453},
  url = {https://arxiv.org/abs/2302.08453},
  author = {Mou, Chong and Wang, Xintao and Xie, Liangbin and Zhang, Jian and Qi, Zhongang and Shan, Ying and Qie, Xiaohu},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), Multimedia (cs.MM), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
