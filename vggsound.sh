#python main.py --train --ckpt_path results/vggsound/normal_frame3 --alpha 0.1 --dataset VGGSound --modulation Normal --pe 0 --gpu_ids 0



 export CUDA_VISIBLE_DEVICES=0
#python main_auxi.py --ckpt_path ./results/ks/full_normal --modality full --dataset  VGGSound --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --drop 0 --gamma 3.0 52.8
#python main_auxi.py --ckpt_path ./results/ks/full_normal --modality full --dataset  VGGSound --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-6 --drop 0 --gamma 3.0  51.9
#python main_auxi.py --ckpt_path ./results/ks/full_normal --modality full --dataset  VGGSound --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --drop 0 --gamma 3.0 --batch_size 128 --learning_rate 2e-3

#python main_auxi.py --ckpt_path ./results/ks/full_normal --modality full --dataset  VGGSound --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-6 --drop 0 --gamma 2.0  51.8

#python main_auxi.py --ckpt_path ./results/ks/full_normal --modality full --dataset  VGGSound --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --drop 0 --gamma 4.0  52.3
#python main_auxi.py --ckpt_path ./results/ks/full_normal --modality full --dataset  VGGSound --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --drop 0 --gamma 4.0  51.5


#python main_auxi_weight.py --ckpt_path ./results/ks/full_normal --modality full --dataset  VGGSound --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --drop 0 --gamma 5.0 52.5


#python main_auxi_weight.py --ckpt_path ./results/ks/full_normal --modality full --dataset  VGGSound --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --drop 0 --gamma 5.0 52.3


# 30 60 90
#python main_auxi_weight.py --ckpt_path ./results/ks/full_normal --modality full --dataset  VGGSound --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --drop 0 --gamma 5.0 54.0

python main_auxi_weight.py --ckpt_path ./results/vggsoud/pe_arl_g_5 --modality full --dataset  VGGSound --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --drop 0 --gamma 5.0