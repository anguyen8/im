METHOD=$1
METRIC=$2
TASK_NAME=$3
OVERWRITE=$4

params=()

# Since SST-2 model is used for SST task, we need to convert it to SST-2 for loading model first and then
# changing it back to SST task below if the flag is on.
if [[ ${TASK_NAME} == "SST" ]]; then
  TASK_NAME="SST-2"
  params+=(--sst_flag)
fi

if [[ ${OVERWRITE} == "--overwrite" ]]; then
  params+=(--overwrite_results)
fi

MODEL_NAME_OR_PATH=../data/models/bert-base-uncased/${TASK_NAME}
DATA_DIR=../data/datasets/${TASK_NAME}
MODEL_BASE=bert-base-uncased
CHECKPOINT=final_dev

params+=(--model_name_or_path "${MODEL_NAME_OR_PATH}")
params+=(--task_name "${TASK_NAME}")
params+=(--do_eval)
params+=(--data_dir "${DATA_DIR}")
params+=(--max_seq_length 128)
params+=(--per_device_eval_batch_size 32)
params+=(--model_base "${MODEL_BASE}")
params+=(--masked_lm "${MODEL_BASE}")
params+=(--analyzer "${METHOD}")
params+=(--checkpoint "${CHECKPOINT}")
params+=(--eval_metric "${METRIC}")

python ../src/run_analyzers.py "${params[@]}"

#python ../src/run_analyzers.py \
#--model_name_or_path ../data/models/bert-base-uncased/${TASK_NAME} \
#--task_name ${TASK_NAME} \
#--sst_flag ${SST_FLAG} \
#--do_eval \
#--data_dir ../data/datasets/${TASK_NAME} \
#--max_seq_length 128 \
#--per_device_eval_batch_size 32 \
#--model_base bert-base-uncased \
#--masked_lm bert-base-uncased \
#--analyzer ${METHOD} \
#--checkpoint final_dev

