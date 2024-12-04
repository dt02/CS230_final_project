source /home/users/deantran/miniconda3/bin/activate
conda activate /scratch/users/deantran/rssl_ppmi/envs

JOB_NAME="eval_rsslt1_adni_sex"
python /scratch/users/deantran/CS230/src/run.py \
    --job_name ${JOB_NAME} \
    --run_mode eval \
    --save_dir /scratch/users/deantran/CS230/experiments/eval_metrics \
    --data_path /scratch/groups/eadeli/data/stru/t1/adni \
    --train_dataset adni \
    --use_amp \
    --num_workers 2 \
    --batch_size 1 \
    --num_levels_for_unet 4 \
    --lr 1e-4 \
    --visualize \
    --label_name sex \
    --max_random_affine_augment_params 0.1 0.1 0.25 0.1 \
    --pretrained_load_path /scratch/users/deantran/CS230/best_models/rsslt1_sex_seed14_epoch10.pth.tar \
    --seed 23