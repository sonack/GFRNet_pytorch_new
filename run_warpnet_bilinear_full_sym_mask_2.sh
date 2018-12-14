# python train_warpnet.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --max_epoch 80 --exp_name "exp_1"

# python train_warpnet_bilinear_full_sym_mask_2.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --train_sym_dir "./DataSets/Original/Sym_NM" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --test_sym_dir "./DataSets/Original/Sym_NM" --max_epoch 100 --exp_name "exp_41" --save_epoch_freq 20 --train_mask_dir "./DataSets/Original/Masks/Intersect" --test_mask_dir "./DataSets/Original/Masks/Intersect"  --face_masks_dir "./DataSets/Original/Masks/Basic" --disp_freq 100 --print_freq 100

python train_warpnet_bilinear_full_sym_mask_2.py --cuda --train_img_dir "./DataSets/Original/Train" --train_landmark_dir "./DataSets/Original/Landmark" --train_sym_dir "./DataSets/Original/Sym_NM" --test_img_dir "./DataSets/Original/Test/testvgg" --test_landmark_dir "./DataSets/Original/Landmark" --test_sym_dir "./DataSets/Original/Sym_NM" --max_epoch 100 --exp_name "exp_41_l1" --save_epoch_freq 20 --train_mask_dir "./DataSets/Original/Masks/Intersect" --test_mask_dir "./DataSets/Original/Masks/Intersect"  --face_masks_dir "./DataSets/Original/Masks/Basic" --disp_freq 100 --print_freq 100 --f2f_kind "l1"
# --load_checkpoint "./checkpoints/exp_40/ckpt_060.pt" 
# --load_checkpoint "./checkpoints/exp_30/ckpt_025.pt"