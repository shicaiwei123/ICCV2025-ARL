
 export CUDA_VISIBLE_DEVICES=0
python main.py --train --ckpt_path results/cramed/audio --alpha 0.1 --modulation Normal --pe 0 --gpu_ids 1 --modality audio