# MPAN BI model (x2) 1 is original 2 is modified by me success2
#python main.py --template MPAN --save MPAN --scale 2 --reset --save_results --patch_size 96 --pre_train ../experiment/model/RCAN_BIX2.pt
#python main.py --template MPAN --save MPANx2 --scale 2 --reset --save_results --patch_size 96 --pre_train ../experiment/MPAN/MPAN_BIX2.pt

# MPAN BI model (x3) success2
#python main.py --template MPAN --save MPANx3 --scale 3 --reset --save_results --patch_size 144 --pre_train ../experiment/model/RCAN_BIX3.pt
#python main.py --template MPAN --save MPANx3_Addition --scale 3 --reset --save_results --patch_size 96 --pre_train ../experiment/model/RCAN_BIX3.pt

# MPAN BI model (x4) success2
#python main.py --template MPAN --save MPANx4 --scale 4 --reset --save_results --patch_size 192 --pre_train ../experiment/model/RCAN_BIX4.pt
#python main.py --template MPAN --save MPANx4_Addition --scale 4 --reset --save_results --patch_size 96 --pre_train ../experiment/model/RCAN_BIX4.pt

# MPAN BI model (x8)
#python main.py --template MPAN --save MPANx8 --scale 8 --reset --save_results --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt
#python main.py --template MPAN --save MPANx8 --scale 8 --reset --save_results --patch_size 96 --pre_train ../experiment/model/RCAN_BIX8.pt
#python main.py --model MPAN --save MPANx8 --scale 8 --reset --n_resgroups 20 --n_resblocks 10 --n_feats 64 --save_results --patch_size 96


# Test MPAN
#python main.py --template MPAN --data_test Set5+Set14+B100+Urban100+Manga109 --data_range 801-900 --scale 2 --pre_train ../experiment/MPAN/MPAN_BIX2.pt --test_only --save MPANx2_test --save_results

