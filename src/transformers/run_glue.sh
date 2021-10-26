export DATA_DIR=../../data/datasets/
export MODEL_BASE=bert-base-uncased
export CHECKPOINT_TRAIN=final_train
export CHECKPOINT_EVAL=final_dev

function train() {

  local TASK_NAME=$1
  local OUTPUT_DIR=$2
  local GPU=$3
  local SEED=$4
  local MODEL_BASE=$5
  local ROAR_TYPE=$6
  local ANALYZER=$7
  local ROAR_RATE=$8

  local ROAR_SETTINGS=${ANALYZER}/${ROAR_RATE}
  local OUTPUT_DIR=${OUTPUT_DIR}/${MODEL_BASE}/${TASK_NAME}/finetuned_${ROAR_TYPE}/${SEED}/${ROAR_SETTINGS}

  mkdir -p $OUTPUT_DIR
  export CUDA_VISIBLE_DEVICES=$GPU

  local TRAIN_DIR=../../data/results/${MODEL_BASE}/${TASK_NAME}/${CHECKPOINT_TRAIN}/${ANALYZER}/roar_examples
  local EVAL_DIR=../../data/results/${MODEL_BASE}/${TASK_NAME}/${CHECKPOINT_EVAL}/${ANALYZER}/roar_examples

  local params=()
  params+=(--do_train)
  params+=(--save_steps 10000)

  params+=(--model_name_or_path "${MODEL_BASE}")
  params+=(--task_name "${TASK_NAME}")
  params+=(--do_eval)
  params+=(--data_dir "${DATA_DIR}/${TASK_NAME}")
  params+=(--max_seq_length 128)
  params+=(--per_device_train_batch_size 32)
  params+=(--per_device_eval_batch_size 32)
  params+=(--gradient_accumulation_steps 1) # Effective bs = per_device_train_batch_size * gradient_accumulation_steps
  params+=(--learning_rate 2e-5)
  params+=(--num_train_epochs 3.0)
  params+=(--output_dir "${OUTPUT_DIR}")
  params+=(--overwrite_output_dir)
  params+=(--overwrite_cache)
  params+=(--seed "${SEED}")

  # FOR ROAR
  params+=(--roar)
  params+=(--attribution_train_dir "${TRAIN_DIR}")
  params+=(--attribution_eval_dir "${EVAL_DIR}")
  params+=(--roar_settings "${ROAR_SETTINGS}")
  params+=(--roar_type "${ROAR_TYPE}")

  echo "${params[@]}"
  python run_glue.py "${params[@]}"
}

function start_training() {
  echo "*** FINETUNING STARTED ***"

  local GPU=0
  local TASK=SST-2
  local OUTPUT_PREFIX=../../data/models

  local SEEDS=(42 100 200 300 400)
  local ROAR_TYPES=(vanilla vanilla_baseline bert bert_baseline)
  local ANALYZERS=(IM LOOEmpty)
  local ROAR_RATES=(0.1 0.2 0.3)

  for i in "${!ROAR_TYPES[@]}"; do
    for j in "${!SEEDS[@]}"; do
      for k in "${!ANALYZERS[@]}"; do
        for l in "${!ROAR_RATES[@]}"; do
          train ${TASK} ${OUTPUT_PREFIX} ${GPU} ${SEEDS[$j]} ${MODEL_BASE} ${ROAR_TYPES[$i]} ${ANALYZERS[$k]} ${ROAR_RATES[$l]}
        done
      done
    done
  done

  echo "*** FINETUNING FINISHED ***"
}

start_training



