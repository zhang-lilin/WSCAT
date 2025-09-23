# Weakly Supervised Contrastive Adversarial Training for Learning Robust Features from Semi-supervised Data

Official implementation of [Weakly Supervised Contrastive Adversarial Training for Learning Robust Features from 
Semi-supervised Data](https://arxiv.org/abs/2503.11032), accepted by CVPR 2025. 

## Requirements

### Dependencies 
```markdown
python==3.10
coloredlogs==15.0.1
matplotlib==3.10.6
numpy==2.3.2
pandas==2.3.2
Pillow==11.3.0
PyYAML==6.0.2
torch==2.6.0+cu126
torchattacks==3.5.1
torchvision==0.21.0+cu126
tqdm==4.67.1
```

### Verified working hardware environment

- Ubuntu 20.04.6 LTS
- CUDA 12.6
- NVIDIA RTX 3090 Ti


## Example usage
```
python tunner.py
```

## Citation
If you find this code useful for your research, please consider citing the following paper:

```markdown
@InProceedings{Zhang_2025_CVPR,
    author    = {Zhang, Lilin and Wu, Chengpei and Yang, Ning},
    title     = {Weakly Supervised Contrastive Adversarial Training for Learning Robust Features from Semi-supervised Data},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {25718-25727}
}
```

