## MPAN

> PyTorch code for our paper "Muti-phase attention network for face super-resolution"
>
> This repository is for MPAN introduced in the paper



> ### Introduction
>
>  Previous general super-resolution methods do not perform well in restoring the details structure information of face images. Prior and attribute-based face super-resolution methods have improved performance with extra trained results. However, they need an additional network and extra training data are challenging to obtain. To address these issues, we propose a Multi-phase Attention Network (MPAN). Specifically, our proposed MPAN builds on integrated residual attention groups (IRAG) and a concatenated attention module (CAM). The IRAG consists of residual channel attention blocks (RCAB) and an integrated attention module (IAM). Meanwhile, we use IRAG to bootstrap the face structures. We utilize the CAM to concentrate on informative layers, hence improving the network's ability to reconstruct facial texture features. We use the IAM to focus on important positions and channels, which makes the network more effective at restoring key face structures like eyes and mouths. The above two attention modules form the multi-phase attention mechanism.
>
>
> Train
> Prepare training data
> Download face images from FFHQ dataset.
>
> ### Begin to train
>
> ```python
> # MPAN BI model (x2) 1 is original 2 is modified by me success2
> #python main.py --template MPAN --save MPAN --scale 2 --reset --save_results --patch_size 96 --pre_train ../experiment/model/RCAN_BIX2.pt
> #python main.py --template MPAN --save MPANx2 --scale 2 --reset --save_results --patch_size 96 --pre_train ../experiment/MPAN/MPAN_BIX2.pt
>
> # MPAN BI model (x3) success2
> #python main.py --template MPAN --save MPANx3 --scale 3 --reset --save_results --patch_size 144 --pre_train ../experiment/model/RCAN_BIX3.pt
> #python main.py --template MPAN --save MPANx3_Addition --scale 3 --reset --save_results --patch_size 96 --pre_train ../experiment/model/RCAN_BIX3.pt
> 
> # MPAN BI model (x4) success2
> #python main.py --template MPAN --save MPANx4 --scale 4 --reset --save_results --patch_size 192 --pre_train ../experiment/model/RCAN_BIX4.pt
> #python main.py --template MPAN --save MPANx4_Addition --scale 4 --reset --save_results --patch_size 96 --pre_train ../experiment/model/RCAN_BIX4.pt
>
> # MPAN BI model (x8)
> #python main.py --template MPAN --save MPANx8 --scale 8 --reset --save_results --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt
> #python main.py --template MPAN --save MPANx8 --scale 8 --reset --save_results --patch_size 96 --pre_train ../experiment/model/RCAN_BIX8.pt
> #python main.py --model MPAN --save MPANx8 --scale 8 --reset --n_resgroups 20 --n_resblocks 10 --n_feats 64 --save_results --patch_size 96
> 
> ```
> 
>
> ###Begin to Test
>
>```
> #test
> # Test MPAN
> #python main.py --template MPAN --data_test Set5+Set14+B100+Urban100+Manga109 --data_range 801-900 --scale 2 --pre_train ../experiment/MPAN/MPAN_BIX2.pt --test_only --save MPANx2_test --save_results
> 
>```
> The whole test pipeline 
>
> 1.Prepare test data.
>
> Run 'Prepare_TestData_HR_LR.m' in Matlab to generate HR/LR images with different degradation models.
>
> 2.Conduct image SR.
>
> See Quick start
>
> 3.Evaluate the results.
>
> Run 'Evaluate_PSNR_SSIM.m' to obtain PSNR/SSIM values for paper.


