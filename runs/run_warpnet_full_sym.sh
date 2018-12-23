# python train_warpnet.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 80 --exp_name "exp_1"

python train_warpnet_full_sym.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --train_sym_dir "./DataSets/Original/Sym_NM" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --test_sym_dir "./DataSets/Original/Sym_NM" --max_epoch 100 --exp_name "exp_30_cont" --save_epoch_freq 20 --load_checkpoint "./checkpoints/exp_30/ckpt_025.pt"