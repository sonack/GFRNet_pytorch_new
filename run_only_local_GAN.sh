python train_only_local_GAN.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_6" --print_freq 10 --disp_freq 10 --load_warpnet "./checkpoints/exp_1/ckpt_020.pt"
# --load_checkpoint "checkpoints/exp_2/ckpt_080.pt"