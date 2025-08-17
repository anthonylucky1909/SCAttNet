# 🚀 SCAttNet: Dual Attention Network for Image Super-Resolution 🔍✨  

[![Hugging Face Spaces][hf-badge]][hf-url]
[![GitHub issues][issues-badge]][issues-url]
[![GitHub stars][stars-badge]][stars-url]
[![License: MIT][license-badge]][license-url]
[![Python Version][python-badge]][python-url]
[![Conda][conda-badge]][conda-url]

## 🖥️ Try Online Demo  

You can try **SCAttNet** directly in your browser without installation:  
👉 [**Live Demo on Hugging Face Spaces**](https://huggingface.co/spaces/anthonyhuang1909/SCAttNet)

> ⚠️ **Note:** The demo is hosted on a free CPU tier at Hugging Face Spaces.  
> Image generation may take some time depending on server load. Please be patient while the results are processed.  

---

_✨ State-of-the-art super-resolution with **dual attention mechanisms** (channel + spatial) for sharper, high-fidelity images._

[![](gif/super_resolution_demo_1.gif)](https://huggingface.co/spaces/anthonyhuang1909/SCAttNet)

## [Live Demo 💥](https://huggingface.co/spaces/anthonyhuang1909/SCAttNet)

---

## 📥 Installation

```bash
conda create -n dualsr python=3.9 -y
conda activate dualsr
pip install -r requirements.txt
```

---

## 🌟 Features

- 🧠 **Dual Attention (Channel + Spatial)** — preserves fine textures and details  
- ⚡ **Progressive Upsampling** — lightweight 4× super-resolution pipeline  
- 🎯 **Flexible Dataset Support** — DIV2K, Urban100, Manga109, BSD100  
- 📊 **Comprehensive Evaluation** — PSNR, SSIM, VGG perceptual loss  
- 🚀 **Simple Training Flow** — run training with one command  

---

## 🔧 Quick Start

### Download datasets
```bash
python download_data.py
```

### Train model
```bash
python main.py
```

### Test on custom image
```python
from model import SuperResolutionNet

model = SuperResolutionNet().load_from_checkpoint("model.pth")
enhanced = model.upscale("your_image.jpg")
```

---

## 🏗️ Architecture Overview

![Model Architecture](model.png)

| Component                         | Description                                                      |
|----------------------------------|------------------------------------------------------------------|
| 🔄 Residual Attention Blocks      | Residual learning combined with attention for robust mapping     |
| 🎛️ Channel-Spatial Attention     | Fusion of channel + spatial attention for adaptive enhancement   |
| 🏗️ Learned Skip Connections       | Trained skip connections to stabilize training                   |
| ⚖️ GAN-Based Training             | Adversarial training for realism and sharpness                   |

---

## 📈 Benchmarks

| Dataset     | Bicubic | SRCNN  | MemNet | EDSR  | RDN   | RCAN  | ESRGAN | **SCAttNet** |
|-------------|---------|--------|--------|-------|-------|-------|--------|--------------|
| **Set14**   | 26.00/0.7027 | 27.50/0.7513 | 28.26/0.7723 | 28.80/0.7876 | 28.81/0.7871 | 28.87/0.7889 | 28.88/0.7896 | **29.18/0.7712** |
| **BSD100**  | 25.96/0.6675 | 26.90/0.7101 | 27.40/0.7281 | 27.71/0.7420 | 27.72/0.7419 | 27.77/0.7436 | 27.76/0.7432 | **28.71/0.7015** |
| **Urban100**| 23.14/0.6577 | 24.52/0.7221 | 25.50/0.7630 | 26.64/0.8033 | 26.61/0.8028 | 26.82/0.8087 | 26.73/0.8072 | **27.99/0.7209** |
| **Manga109**| 24.89/0.7866 | 27.58/0.8555 | 29.42/0.8942 | 31.02/0.9148 | 31.00/0.9151 | 31.22/0.9173 | 31.16/0.9164 | **27.33/0.7765** |

---

## 📚 Citation

```bibtex
@misc{huang2025scattnet,
  author = {Anthony Huang},
  title = {SCAttNet: Dual Attention Network for Image Super-Resolution},
  year = {2025},
  howpublished = {Hugging Face Spaces},
  url = {https://huggingface.co/spaces/anthonyhuang1909/SCAttNet}
}
```

---

## 🤝 Contributing

All contributions are welcome! 🎉  
- Fork the repo  
- Create a feature branch  
- Submit a PR 🚀  

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

<!-- Badge References -->
[hf-badge]: https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue
[hf-url]: https://huggingface.co/spaces/anthonyhuang1909/SCAttNet

[issues-badge]: https://img.shields.io/github/issues/anthonyhuang1909/SCAttNet
[issues-url]: https://github.com/anthonyhuang1909/SCAttNet/issues

[stars-badge]: https://img.shields.io/github/stars/anthonyhuang1909/SCAttNet
[stars-url]: https://github.com/anthonyhuang1909/SCAttNet/stargazers

[license-badge]: https://img.shields.io/badge/License-MIT-green.svg
[license-url]: https://opensource.org/licenses/MIT

[python-badge]: https://img.shields.io/badge/python-3.9%2B-blue
[python-url]: https://www.python.org/

[conda-badge]: https://img.shields.io/badge/conda-ready-brightgreen
[conda-url]: https://docs.conda.io/

[ci-badge]: https://github.com/anthonyhuang1909/SCAttNet/actions/workflows/ci.yml/badge.svg
[ci-url]: https://github.com/anthonyhuang1909/SCAttNet/actions

[tests-badge]: https://github.com/anthonyhuang1909/SCAttNet/actions/workflows/tests.yml/badge.svg
[tests-url]: https://github.com/anthonyhuang1909/SCAttNet/actions

[coverage-badge]: https://img.shields.io/codecov/c/github/anthonyhuang1909/SCAttNet
[coverage-url]: https://codecov.io/gh/anthonyhuang1909/SCAttNet
