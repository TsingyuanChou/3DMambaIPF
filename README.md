<h2 align="center"> <a href="https://ojs.aaai.org/index.php/AAAI/article/view/33178">3DMambaIPF: A State Space Model for Iterative  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/33178"> Point Cloud Filtering via Differentiable Rendering </a>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2404.05522-b31b1b.svg?logo=arXiv)](https://arxiv.org/2404.05522)
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://3DMambaIPF.github.io/)
</h5>

Qingyuan Zhou, Weidong Yang, Ben Fei, Jingyi Xu, Rui Zhang, Keyi Liu, Yeqi Luo, Ying He

Paper: https://ojs.aaai.org/index.php/AAAI/article/view/33178

Paper (Preprint and Suppl.): 

Abstract: Noise is an inevitable aspect of point cloud acquisition, necessitating filtering as a fundamental task within the realm of 3D vision. Existing learning-based filtering methods have shown promising capabilities on commonly used datasets. Nonetheless, the effectiveness of these methods is constrained when dealing with a substantial quantity of point clouds. This limitation primarily stems from their limited denoising capabilities for dense and large-scale point clouds and their inclination to generate noisy outliers after denoising. To deal with this challenge, we introduce 3DMambaIPF, for the first time, exploiting Selective State Space Models (SSMs) architecture to handle highly-dense and large-scale point clouds, capitalizing on its strengths in selective input processing and large context modeling capabilities. Additionally, we present a robust and fast differentiable rendering loss to constrain the noisy points around the surface. In contrast to previous methodologies, this differentiable rendering loss enhances the visual realism of denoised geometric structures and aligns point cloud boundaries more closely with those observed in real-world objects. Extensive evaluations on commonly used datasets (typically with up to 50K points) demonstrate that 3DMambaIPF achieves state-of-the-art results. Moreover, we showcase the superior scalability and efficiency of 3DMambaIPF on highly dense and large-scale point clouds with up to 500K points compared to off-the-shelf methods.

## Installation
```
conda create -n mbipf python=3.9
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pytorch-cluster -c pyg
pip install mamba-ssm == 1.1.3.post1
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pyg -c pyg
conda install pytorch3d -c pytorch3d
conda install pytorch-lightning
pip install torch-scatter
pip install point-cloud-utils
pip install plyfile
pip install pandas
pip install tensorboard
```

## Pretrained Model
[Download here](https://drive.google.com/file/d/11VJMq4zH56eWIaAe9YGvB8g9YLA9k35M/view?usp=sharing)
