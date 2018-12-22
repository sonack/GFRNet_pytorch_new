# python train_warpnet.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 80 --exp_name "exp_1"

# python train_save_warpnet_results.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --train_sym_dir "./DataSets/Original/Sym_NM" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --test_sym_dir "./DataSets/Original/Sym_NM" --max_epoch 100 --exp_name "exp_40_swt_testvgg" --save_epoch_freq 20 --load_checkpoint "./checkpoints/exp_1/ckpt_040.pt" --load_checkpoint_B "./checkpoints/exp_31/ckpt_100.pt" --load_checkpoint_C "./checkpoints/exp_40/ckpt_080.pt"

python train_save_warpnet_results.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --train_sym_dir "./DataSets/Original/Sym_NM" --test_img_dir "./DataSets/Original/Test/test_small" --test_landmark_dir "./DataSets/Original/Landmark" --test_sym_dir "./DataSets/Original/Sym_NM" --max_epoch 100 --exp_name "exp_40_swt_small" --save_epoch_freq 20 --load_checkpoint "./checkpoints/exp_1/ckpt_040.pt" --load_checkpoint_B "./checkpoints/exp_31/ckpt_100.pt" --load_checkpoint_C "./checkpoints/exp_40/ckpt_080.pt"


# --load_checkpoint "./checkpoints/exp_1/ckpt_040.pt" --load_checkpoint_B "./checkpoints/exp_30_cont/ckpt_100.pt" --load_checkpoint_C "./checkpoints/exp_31/ckpt_100.pt"


# exp_30 round
# exp_31 bilinear
# exp_40 mask_1
