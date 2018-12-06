# python train_full_local_GAN.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "exp_7" --print_freq 100 --disp_freq 100 --load_checkpoint "checkpoints/exp_4/ckpt_100.pt"
# --load_warpnet "./checkpoints/exp_1/ckpt_020.pt"


# st save test results images out 
# python train_save_results.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "st_1_2" --print_freq 100 --disp_freq 100 --load_checkpoint './checkpoints/exp_13_3/ckpt_245.pt' --kind 'weaker_1' --load_sbt_dir "./sbt/sb_7"

python train_save_results.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 1500 --exp_name "st_sb_7" --print_freq 100 --disp_freq 100 --load_checkpoint './checkpoints/exp_13_3/ckpt_245.pt' --kind 'weaker_1' --load_sbt_dir "./sbt/sb_7"