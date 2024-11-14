source /home/users/deantran/miniconda3/bin/activate
conda activate /scratch/users/deantran/rssl_ppmi/envs

JOB_NAME="eval_rsslt1_adni_group"
python /scratch/users/deantran/CS230/src/run.py \
    --job_name ${JOB_NAME} \
    --run_mode eval \
    --save_dir /scratch/users/deantran/CS230/experiments/unet_adni_finetune \
    --data_path /scratch/groups/eadeli/data/stru/t1/adni \
    --train_dataset adni \
    --use_amp \
    --num_workers 2 \
    --batch_size 1 \
    --num_levels_for_unet 4 \
    --epochs 150000 \
    --steps_per_epoch \
    --lr 1e-4 \
    --visualize \
    --label_name group \
    --max_random_affine_augment_params 0.1 0.1 0.25 0.1 \
    --pretrained_load_path /scratch/users/deantran/CS230/experiments/unet_adni_finetune/train_rsslt1_adni_group_dataadni_batch1_lr0.0001/checkpoints/epoch1_trained_model.pth.tar \
    --seed 23