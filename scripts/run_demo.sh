TASK_NAME=$1
TEXT_A=$2
TEXT_B=$3
THETA=$4

params=()

# Since SST-2 model is used for SST task, we need to convert it to SST-2 for loading model first and then
# changing it back to SST task below if the flag is on.
if [[ ${TASK_NAME} == "SST" ]]; then
  TASK_NAME="SST-2"
  params+=(--sst_flag)
fi

MODEL_NAME_OR_PATH=../data/models/bert-base-uncased/${TASK_NAME}
DATA_DIR=../data/datasets/${TASK_NAME}
MODEL_BASE=bert-base-uncased

if [[ ${TASK_NAME} == "SST" || ${TASK_NAME} == "SST-2" ]]; then
  MODEL_NAME_OR_PATH=pmthangk09/bert-base-uncased-glue-sst2
elif [[ ${TASK_NAME} == "ESNLI" ]]; then
  MODEL_NAME_OR_PATH=pmthangk09/bert-base-uncased-esnli
elif [[ ${TASK_NAME} == "ESNLI" ]]; then
  MODEL_NAME_OR_PATH=pmthangk09/bert-base-uncased-superglue-multirc
fi

params+=(--model_name_or_path "${MODEL_NAME_OR_PATH}")
params+=(--task_name "${TASK_NAME}")
params+=(--do_eval)
params+=(--data_dir "${DATA_DIR}")
params+=(--max_seq_length 128)
params+=(--per_device_eval_batch_size 32)
params+=(--model_base "${MODEL_BASE}")
params+=(--masked_lm "${MODEL_BASE}")
params+=(--text_a "${TEXT_A}")
params+=(--text_b "${TEXT_B}")
params+=(--theta "${THETA}")

python run_demo.py "${params[@]}"


