TASK_NAME=$1
METHOD=$2

python ../src/run_analyzers.py \
--model_name_or_path ../data/models/bert-base-uncased/${TASK_NAME} \
--task_name ${TASK_NAME} \
--do_eval \
--data_dir ../data/datasets/${TASK_NAME} \
--max_seq_length 128 \
--per_device_eval_batch_size 32 \
--model_base bert-base-uncased \
--masked_lm bert-base-uncased \
--analyzer ${METHOD} \
--output_dir ../data/models/bert-base-uncased/${TASK_NAME}/debug \
--overwrite_output_dir \
--checkpoint sanity_check_final_dev \
--seed 100