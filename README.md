# 3DMambaIPF: A State Space Model for Iterative Point Cloud Filtering via Differentiable Rendering
Qingyuan Zhou, Weidong Yang, Ben Fei, Jingyi Xu, Rui Zhang, Keyi Liu, Yeqi Luo, Ying He

Paper: https://arxiv.org/abs/2404.05522

Abstract: Noise is an inevitable aspect of point cloud acquisition, necessitating filtering as a fundamental task within the realm of 3D vision. Existing learning-based filtering methods have shown promising capabilities on commonly used datasets. Nonetheless, the effectiveness of these methods is constrained when dealing with a substantial quantity of point clouds. This limitation primarily stems from their limited denoising capabilities for dense and large-scale point clouds and their inclination to generate noisy outliers after denoising. To deal with this challenge, we introduce 3DMambaIPF, for the first time, exploiting Selective State Space Models (SSMs) architecture to handle highly-dense and large-scale point clouds, capitalizing on its strengths in selective input processing and large context modeling capabilities. Additionally, we present a robust and fast differentiable rendering loss to constrain the noisy points around the surface. In contrast to previous methodologies, this differentiable rendering loss enhances the visual realism of denoised geometric structures and aligns point cloud boundaries more closely with those observed in real-world objects. Extensive evaluations on commonly used datasets (typically with up to 50K points) demonstrate that 3DMambaIPF achieves state-of-the-art results. Moreover, we showcase the superior scalability and efficiency of 3DMambaIPF on highly dense and large-scale point clouds with up to 500K points compared to off-the-shelf methods.
