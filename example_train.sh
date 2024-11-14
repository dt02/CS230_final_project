source /home/users/deantran/miniconda3/bin/activate
conda activate /scratch/users/deantran/rssl_ppmi/envs

JOB_NAME="train_rsslt1_adni_group"
python /scratch/users/deantran/CS230/run.py \
    --job_name ${JOB_NAME} \
    --run_mode eval \
    --wandb_kwargs project=CS230 name=$JOB_NAME \
    --save_dir /scratch/users/deantran/CS230/experiments/unet_adni_finetune \
    --data_path /scratch/groups/eadeli/data/stru/t1/adni \
    --train_dataset adni \
    --use_amp \
    --num_workers 2 \
    --batch_size 1 \
    --num_levels_for_unet 4 \
    --epochs 150000 \
    --steps_per_epoch 10 \
    --lr 1e-4 \
    --visualize \
    --label_name group \
    --max_random_affine_augment_params 0.1 0.1 0.25 0.1 \
    --pretrained_load_path /scratch/users/deantran/rssl_ppmi/pretrained_models/rsslt1_pretrained_epoch2100_model.pth.tar \
    --seed 23