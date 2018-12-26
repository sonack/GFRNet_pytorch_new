# python train_only_local_GAN.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_6" --print_freq 100 --disp_freq 100 --load_warpnet "./checkpoints/exp_1/ckpt_020.pt"
# --load_checkpoint "checkpoints/exp_2/ckpt_080.pt"

# python train_only_local_GAN.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_6_debug_grad" --print_freq 10 --disp_freq 10 --load_warpnet "./checkpoints/exp_1/ckpt_020.pt"
# --load_checkpoint "checkpoints/exp_2/ckpt_080.pt"


# python train_full_part_GAN.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_11" --print_freq 100 --disp_freq 100 --load_warpnet "./checkpoints/exp_1/ckpt_040.pt" --load_checkpoint './checkpoints/exp_7/ckpt_180.pt'

# python train_full_part_local_GAN.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_12_2" --print_freq 100 --disp_freq 100 --load_checkpoint './checkpoints/exp_12/ckpt_215.pt' --save_epoch_freq 5 --pd_L_l_w 0.5 --pd_R_l_w 0.5 --pd_N_l_w 0.5 --pd_M_l_w 0.5

# python train_full_part_local_GAN.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_13" --print_freq 100 --disp_freq 100 --load_checkpoint './checkpoints/exp_12_2/ckpt_220.pt' --save_epoch_freq 10 --pd_L_l_w 1.5 --pd_R_l_w 1.5 --pd_N_l_w 1.5 --pd_M_l_w 1.5 --gd_l_w 2 --ld_l_w 1

# python train_full_part_local_GAN.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_13_2" --print_freq 100 --disp_freq 100 --load_checkpoint './checkpoints/exp_13/ckpt_240.pt' --save_epoch_freq 5 --pd_L_l_w 3 --pd_R_l_w 3 --pd_N_l_w 3 --pd_M_l_w 3 --gd_l_w 10 --ld_l_w 5

# python train_full_part_local_GAN.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_13_3" --print_freq 100 --disp_freq 100 --load_checkpoint './checkpoints/exp_13/ckpt_240.pt' --save_epoch_freq 1 --pd_L_l_w 2 --pd_R_l_w 2 --pd_N_l_w 2 --pd_M_l_w 2 --gd_l_w 2 --ld_l_w 1 --kind "weaker_1"

# python train_full_part_local_GAN_cond.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_16" --print_freq 100 --disp_freq 100  --save_epoch_freq 10 --pd_L_l_w 100 --pd_R_l_w 100 --pd_N_l_w 100 --pd_M_l_w 100 --gd_l_w 200 --ld_l_w 100 --kind "weaker_1" --GD_cond 6 --PD_cond 6 --load_checkpoint './checkpoints/exp_13_3/ckpt_245.pt'


# python train_full_part_local_GAN_cond.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_17" --print_freq 100 --disp_freq 100  --save_epoch_freq 10 --pd_L_l_w 100 --pd_R_l_w 100 --pd_N_l_w 100 --pd_M_l_w 100 --gd_l_w 1 --ld_l_w 0.5 --kind "weaker_1" --GD_cond 6 --PD_cond 6 --load_checkpoint './checkpoints/exp_16/ckpt_260.pt'

# dataset SP shrink parts set to 0.8
# python train_full_part_local_GAN_cond_SP_LR.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_18" --print_freq 100 --disp_freq 100  --save_epoch_freq 10 --pd_L_l_w 100 --pd_R_l_w 100 --pd_N_l_w 100 --pd_M_l_w 100 --gd_l_w 1 --ld_l_w 0.5 --lr_l_w 10 --kind "weaker_1" --GD_cond 6 --PD_cond 6 --load_checkpoint './checkpoints/exp_17/ckpt_300.pt'

# dataset SP shrink parts set to 1.2
# python train_full_part_local_GAN_cond_SP_LR.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_19" --print_freq 100 --disp_freq 100  --save_epoch_freq 5 --pd_L_l_w 1 --pd_R_l_w 1 --pd_N_l_w 1 --pd_M_l_w 1 --gd_l_w 1 --ld_l_w 0.5 --lr_l_w 1 --kind "weaker_1" --GD_cond 6 --PD_cond 6 --load_checkpoint './checkpoints/exp_18/ckpt_310.pt' --parts_expand 1.2


# python train_wgan.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_43" --print_freq 10 --disp_freq 10  --save_epoch_freq 10 --pd_L_l_w 2 --pd_R_l_w 2 --pd_N_l_w 2 --pd_M_l_w 2 --gd_l_w 1 --ld_l_w 0.5 --lr_l_w 1 --perp_l_w 0 --kind "weaker_1" --GD_cond 6 --PD_cond 6 --parts_expand 1.2 --use_WGAN --load_warpnet "./checkpoints/exp_1/ckpt_040.pt"
# --adam
# --load_checkpoint './checkpoints/exp_19/ckpt_465.pt'



# python train_wgan.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_43_just_wgan" --print_freq 100 --disp_freq 100  --save_epoch_freq 10 --pd_L_l_w 2 --pd_R_l_w 2 --pd_N_l_w 2 --pd_M_l_w 2 --gd_l_w 1 --ld_l_w 0.5 --lr_l_w 1 --perp_l_w 0 --kind "weaker_1" --GD_cond 6 --PD_cond 6 --parts_expand 1.2 --use_WGAN --load_warpnet "./checkpoints/exp_1/ckpt_040.pt"

# python train_wgan.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_44_only_part_lr=2e-5" --print_freq 100 --disp_freq 100 --save_epoch_freq 1 --pd_L_l_w 2 --pd_R_l_w 2 --pd_N_l_w 2 --pd_M_l_w 2 --gd_l_w 1 --ld_l_w 0.5 --lr_l_w 1 --perp_l_w 0 --kind "weaker_1" --GD_cond 6 --PD_cond 6 --parts_expand 1.2 --use_WGAN_GP --load_warpnet "./checkpoints/exp_1/ckpt_040.pt" --lr 2e-5


# python train_wgan.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_45_subpixel_lr_dbg" --print_freq 10 --disp_freq 10 --save_epoch_freq 1 --pd_L_l_w 2 --pd_R_l_w 2 --pd_N_l_w 2 --pd_M_l_w 2 --gd_l_w 1 --ld_l_w 0.5 --lr_l_w 1 --perp_l_w 0 --kind "original" --GD_cond 6 --PD_cond 6 --parts_expand 1.2 --use_WGAN_GP --load_warpnet "./checkpoints/exp_1/ckpt_040.pt" --lr 2e-4 --jpeg_last --deconv_kind "subpixel" --skip_train_D --debug --lr 4e-3

# --mse_l_w 2 
# --debug
# --deconv_kind "subpixel"


# weaker_1 subpixel finetune only mse, no jpeg_last
python train_wgan.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_45_subpixel_ft_weaker1" --print_freq 10 --disp_freq 10 --save_epoch_freq 1 --pd_L_l_w 2 --pd_R_l_w 2 --pd_N_l_w 2 --pd_M_l_w 2 --gd_l_w 1 --ld_l_w 0.5 --lr_l_w 1 --perp_l_w 0 --kind "weaker_1" --GD_cond 6 --PD_cond 6 --parts_expand 1.2 --use_WGAN_GP --deconv_kind "subpixel" --skip_train_D --debug --lr 4e-3 --load_checkpoint "./checkpoints/exp_45_subpixel_lr_dbg/ckpt_050.pt"


# exp_45_test_vgg_conv3_original
# python train_wgan.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_45_test_vgg_conv3_original" --print_freq 10 --disp_freq 10 --save_epoch_freq 1 --pd_L_l_w 2 --pd_R_l_w 2 --pd_N_l_w 2 --pd_M_l_w 2 --gd_l_w 1 --ld_l_w 0.5 --lr_l_w 1 --perp_l_w 0 --kind "original" --GD_cond 6 --PD_cond 6 --parts_expand 1.2 --use_WGAN_GP --load_warpnet "./checkpoints/exp_1/ckpt_040.pt" --jpeg_last --deconv_kind "subpixel" --skip_train_D --debug --lr 4e-3 --load_checkpoint "./checkpoints/exp_45_subpixel_lr_dbg/ckpt_050.pt"
