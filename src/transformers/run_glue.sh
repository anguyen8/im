export GLUE_DIR=/home/home2/thang/Projects/open_sources/transformer_gp2/examples/glue_data


function train() {

  local TASK_NAME=$1
  local OUTPUT_DIR=$2
  local GPU=$3
  local SEED=$4
  local MODEL_BASE=$5
  local SYNTHETIC_TASK=$6

#  local OUTPUT_DIR=${OUTPUT_DIR}/${MODEL_BASE}/${TASK_NAME}/finetuned
#  local OUTPUT_DIR=${OUTPUT_DIR}/${MODEL_BASE}/${TASK_NAME}/rebuttal/finetuned_topN_10_use_bert/${SEED}/${SYNTHETIC_TASK}
  local OUTPUT_DIR=${OUTPUT_DIR}/${MODEL_BASE}/${TASK_NAME}/finetuned_topN_10/${SEED}/${SYNTHETIC_TASK}

  # For synthetic ONLY - BORROW SHUFFLE_DIR as SYNTHETIC_TASK
#  local OUTPUT_DIR=${OUTPUT_DIR}/roberta-synthetic/${SYNTHETIC_TASK}/finetuned
#  local OUTPUT_DIR=${OUTPUT_DIR}/roberta-extra-finetune/${SEED}/${TASK_NAME}/finetuned

  mkdir -p $OUTPUT_DIR
  export CUDA_VISIBLE_DEVICES=$GPU

  local params=()
  params+=(--do_train)
  params+=(--save_steps 10000)

  params+=(--model_name_or_path "${MODEL_BASE}")
  params+=(--task_name "${TASK_NAME}")
  params+=(--do_eval)
  params+=(--data_dir "${GLUE_DIR}/${TASK_NAME}")
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
  params+=(--shuffle_dir "${SYNTHETIC_TASK}") # Use for finetuning on synthetic task only

  echo "${params[@]}"
  nohup python -u text-classification/run_glue.py "${params[@]}" > $OUTPUT_DIR/finetune_logs.txt &
}

function run() {

  local TASK_NAME=$1
  local BASE_DIR=$2
  local GPU=$3
  local SEED=$4
  local MODEL_BASE=$5
  local MODE=$6
  local SHUFFLE_DIR=$7

  # ============================================================================================================
  # Use ONLY for multiple runs purpose
#  local OUTPUT_DIR=${BASE_DIR}/${MODEL_BASE}/${TASK_NAME}/evaluation/${SEED}/${SHUFFLE_DIR}/${MODE}   # For ANLI - SHUFFLE_DIR in [r1, r2, r3]
#  local OUTPUT_DIR=${BASE_DIR}/${MODEL_BASE}/${SEED}/${TASK_NAME}/evaluation/${SHUFFLE_DIR}/${MODE}   # Extra-finetuning for ANLI only - SHUFFLE_DIR in [r1, r2, r3]
#  local OUTPUT_DIR=${BASE_DIR}/${MODEL_BASE}/${SEED}/${TASK_NAME}/evaluation/${MODE}                  # Extra-finetuning
  local OUTPUT_DIR=${BASE_DIR}/${MODEL_BASE}/${TASK_NAME}/evaluation_rebuttal/${SEED}/${MODE}          # ACL rebuttal
#  local OUTPUT_DIR=${BASE_DIR}/${MODEL_BASE}/${TASK_NAME}/evaluation/${SEED}/${MODE}                   # Otherwise

  # Otherwise, use this:
#  local OUTPUT_DIR=${BASE_DIR}/${MODEL_BASE}/${TASK_NAME}/evaluation/${MODE}                          # Normal finetuning
#  local OUTPUT_DIR=${BASE_DIR}/${MODEL_BASE}/${TASK_NAME}/evaluation/${SHUFFLE_DIR}/${MODE}

  # For synthetic ONLY - BORROW SHUFFLE_DIR as SYNTHETIC_TASK
#  local OUTPUT_DIR=${BASE_DIR}/roberta-synthetic/${SHUFFLE_DIR}/evaluation/${MODE}                    # Evaluate synthetic models
  if [[ ${CHECKED_POINT} != "None" ]]; then
    local OUTPUT_DIR=${BASE_DIR}/${MODEL_BASE}/${TASK_NAME}/evaluation/${CHECKED_POINT}/${MODE}
  fi

  echo ${OUTPUT_DIR}
  # ============================================================================================================

  mkdir -p $OUTPUT_DIR
  export CUDA_VISIBLE_DEVICES=$GPU

  local params=()

  if [[ ${MODE} == "shuffled"* ]]; then
    params+=(--shuffle_data)
    params+=(--shuffle_type "${MODE}")
    echo ${OUTPUT_DIR}
  fi

  if [[ ${MODE} == "swapped"* ]]; then
    params+=(--shuffle_data)
    params+=(--shuffle_type "${MODE}")
    echo ${OUTPUT_DIR}
  fi

  # ThangPM: Automatically load pretrained model
  # ============================================================================================================
#  local MODEL_NAME_OR_PATH=${BASE_DIR}/${MODEL_BASE}/${SEED}/${TASK_NAME}/finetuned   # Extra-finetuning
  local MODEL_NAME_OR_PATH=${BASE_DIR}/${MODEL_BASE}/${TASK_NAME}/finetuned           # Normal finetuning
  if [[ ${MODE} == "baseline" ]]; then
    local MODEL_NAME_OR_PATH=${MODEL_BASE}
  fi

  if [[ ${CHECKED_POINT} != "None" ]]; then
    local MODEL_NAME_OR_PATH=${BASE_DIR}/${MODEL_BASE}/${TASK_NAME}/finetuned/${CHECKED_POINT}
  fi

  # For synthetic ONLY - BORROW SHUFFLE_DIR as SYNTHETIC_TASK
#  local MODEL_NAME_OR_PATH=${BASE_DIR}/roberta-synthetic/${SHUFFLE_DIR}/finetuned    # Load synthetic models for evaluation

#  local MODEL_NAME_OR_PATH=ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli       # Hard-coded for ANLI of the original paper
  # ============================================================================================================

  params+=(--model_name_or_path "${MODEL_NAME_OR_PATH}")
  params+=(--task_name "${TASK_NAME}")
  params+=(--do_eval)
#  params+=(--do_predict)
  params+=(--data_dir "${GLUE_DIR}/${TASK_NAME}")
  params+=(--max_seq_length 128)
  params+=(--per_device_eval_batch_size 32)
  params+=(--output_dir "${OUTPUT_DIR}")
  params+=(--overwrite_output_dir)
  params+=(--overwrite_cache)
  params+=(--seed "${SEED}")
#  params+=(--get_sent_embs)  ###### -------------------------------- REMEMBER TO UNCOMMENT THIS FOR MINIQ1
  params+=(--shuffle_dir "${SHUFFLE_DIR}")

  # Use ONLY for multiple runs purpose
   python text-classification/run_glue.py "${params[@]}"
  # Otherwise, use this:
#  nohup python -u text-classification/run_glue.py "${params[@]}" > $OUTPUT_DIR/predict_logs.txt &
}

function train_glue() {

  echo "*** FINETUNING STARTED ***"

  # There are only 8 GPUs so CoLA and SST-2 will share the same GPU_0
  train CoLA models 0 42 ${MODEL_BASE}

  local glue_tasks=(SST-2 MRPC STS-B QQP MNLI QNLI RTE WNLI)
  for i in "${!glue_tasks[@]}"; do
    train ${glue_tasks[$i]} models $i 42 ${MODEL_BASE}
  done
}

function train_single_glue() {

  echo "*** FINETUNING STARTED ***"

  local i=$1
  local GPU=$2
  local synthetic_task=$3
  local SEED=$4

  if [[ ${GPU} -gt 7 ]]; then
    local GPU=0
  fi

  local glue_tasks=(CoLA SST-2 MRPC STS-B QQP MNLI QNLI RTE WNLI)

  # Only for finetuning synthetic model on downstream tasks
#  export MODEL_BASE=models/roberta-synthetic/${glue_tasks[$i]}/finetuned
#  export MODEL_BASE=models/roberta-synthetic/${synthetic_task}/finetuned
#  echo ${MODEL_BASE}

  train ${glue_tasks[$i]} models $GPU ${SEED} ${MODEL_BASE} ${synthetic_task}
}

function run_glue() {

  echo "*** EVALUATION STARTED ***"

  MODE=$1

  # There are only 8 GPUs so CoLA and SST-2 will share the same GPU_0
  run CoLA models 0 42 ${MODEL_BASE} ${MODE}

  local glue_tasks=(SST-2 MRPC STS-B QQP MNLI QNLI RTE WNLI)
  for i in "${!glue_tasks[@]}"; do
    run ${glue_tasks[$i]} models $i 42 ${MODEL_BASE} ${MODE}
  done
}

function run_single_glue() {

  echo "*** EVALUATION STARTED ***"

  local i=$1
  local MODE=$2
  local GPU=$3
  local synthetic_task=$4

  if [[ ${GPU} -gt 7 ]]; then
    local GPU=0
  fi

  local glue_tasks=(CoLA SST-2 MRPC STS-B QQP MNLI QNLI RTE WNLI)

#  run ${glue_tasks[$i]} models $GPU 42 ${MODEL_BASE} ${MODE} roberta_cola
  run ${glue_tasks[$i]} models $GPU 42 ${MODEL_BASE} ${MODE} ${synthetic_task}
}

function run_single_glue_with_seed() {

  echo "*** EVALUATION STARTED ***"

  local i=$1
  local SEED=$2
  local MODE=$3

  local GPU=$4
  local SHUFFLE_DIR=$5

  if [[ ${GPU} -gt 7 ]]; then
    local GPU=0
  fi

  local glue_tasks=(CoLA SST-2 MRPC STS-B QQP MNLI QNLI RTE WNLI)

  run ${glue_tasks[$i]} models $GPU ${SEED} ${MODEL_BASE} ${MODE} ${SHUFFLE_DIR}
}

function run_multiple_times() {
#  local modes=(original shuffled shuffled_bigram shuffled_trigram)
#  local modes=(swapped_nouns)
  local modes=(shuffled_2_sents)

#  local seeds=(422)                                          # Use for normal finetuning
  local seeds=(100 200 300 400 500 600 700 800 900 1000)    # Use for normal finetuning
#  local seeds=(42 100 200 300 400)                          # For extra finetuning

  local GPU=0
#  local SHUFFLE_DIR=10_runs_roberta_large_miniQ1   # GPU 0
#  local SHUFFLE_DIR=10_runs_roberta_miniQ1         # GPU 0
#  local SHUFFLE_DIR=10_runs_albert_miniQ1          # GPU 1
#  local SHUFFLE_DIR=10_runs_bert_miniQ1            # GPU 2
  local SHUFFLE_DIR=10_runs_acl_rebuttal_miniQ1     # RoBERTa: GPU0, ALBERT: GPU1, BERT: GPU2
#  local SHUFFLE_DIR=10_runs_albert_2nd_miniQ1      # GPU 0 (Re-running for RTE and QQP)
#  local SHUFFLE_DIR=roberta_synthetic_adv_glue

  for mode in "${modes[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "$seed --- $mode"
#        run_single_glue_with_seed 0 $seed $mode $GPU $SHUFFLE_DIR   # CoLA
#        run_single_glue_with_seed 1 $seed $mode $GPU $SHUFFLE_DIR   # SST-2
        run_single_glue_with_seed 2 $seed $mode $GPU $SHUFFLE_DIR   # MRPC
#        run_single_glue_with_seed 3 $seed $mode $GPU $SHUFFLE_DIR   # STS-B
        run_single_glue_with_seed 4 $seed $mode $GPU $SHUFFLE_DIR   # QQP
#        run_single_glue_with_seed 6 $seed $mode $GPU $SHUFFLE_DIR   # QNLI
#        run_single_glue_with_seed 7 $seed $mode $GPU $SHUFFLE_DIR   # RTE

#        run_single_glue_with_seed 5 $seed $mode 0 r1
#        run_single_glue_with_seed 5 $seed $mode 1 r2
#        run_single_glue_with_seed 5 $seed $mode 2 r3
    done
  done
}

#export MODEL_BASE=roberta-base
#export MODEL_BASE=albert-base-v2
export MODEL_BASE=bert-base-uncased

#export MODEL_BASE=models/roberta-base/CoLA/finetuned
#export MODEL_BASE=roberta-base-cola
#export MODEL_BASE=roberta-extra-finetune

#export MODEL_BASE=bert-large-uncased
#export MODEL_BASE=roberta-large
#export MODEL_BASE=albert-xxlarge-v2

#export MODEL_BASE=ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli

function start_training() {
  # MODE in [baseline, original, shuffled, shuffled_bigram, shuffled_trigram]
  #train_glue
  #train_single_glue 0 0 None 42

  local SEED=$1

#  train_single_glue 1 0 InputMargin/0.1 ${SEED}
#  train_single_glue 1 1 InputMargin/0.2 ${SEED}
#  train_single_glue 1 2 InputMargin/0.3 ${SEED}
#  train_single_glue 1 1 InputMargin/0.5 ${SEED}
#  train_single_glue 1 2 InputMargin/0.8 ${SEED}

#  train_single_glue 1 5 OccEmpty/0.1 ${SEED}
#  train_single_glue 1 6 OccEmpty/0.2 ${SEED}
  train_single_glue 1 7 OccEmpty/0.3 ${SEED}
#  train_single_glue 1 4 OccEmpty/0.5 ${SEED}
#  train_single_glue 1 5 OccEmpty/0.8 ${SEED}

#  train_single_glue 1 0 original 42
#  train_single_glue 1 1 original 100
#  train_single_glue 1 2 original 200
#  train_single_glue 1 3 original 300
#  train_single_glue 1 6 original 400

  #train_single_glue 2 1 None 42
  #train_single_glue 3 2 None 42
  #train_single_glue 4 2 None 42
  #train_single_glue 6 3 None 42
  #train_single_glue 7 3 None 42

  #train_single_glue 1 4 SST-2 400
  #train_single_glue 2 2 MRPC 500
  #train_single_glue 4 3 QQP 400
  #train_single_glue 6 6 QNLI 400
  #train_single_glue 7 5 RTE 500

  #train_single_glue 0 0 SQuAD_v2 42

  # TASK 3: COME BACK LATER (FINETUNE SYNTHETIC MODEL ON DOWNSTREAM TASK)
  #train_single_glue 3 0 STS-B 42
  #train_single_glue 3 1 STS-B 100
  #train_single_glue 3 2 STS-B 200
  #train_single_glue 3 3 STS-B 300
  #train_single_glue 3 7 STS-B 400

  #train_single_glue 0 1 ANLI 42  # Finetuning on Synthetic task for MNLI/ANLI
  #train_single_glue 5 1 ANLI 42  # Finetuning on Downstream task task for MNLI/ANLI
  #train_single_glue 5 2 ANLI 100
  #train_single_glue 5 3 ANLI 200
  #train_single_glue 5 5 ANLI 300
  #train_single_glue 5 6 ANLI 400
  #train_single_glue 5 7 ANLI 500
}

function start_evaluation() {
  # MODE in [baseline, original, shuffled, shuffled_bigram, shuffled_trigram]
  #run_glue original
  #run_single_glue 3 baseline

  export CHECKED_POINT=None

  #run_single_glue 0 original 0
#  run_single_glue 1 original 0
  #run_single_glue 2 original 1
  #run_single_glue 3 original 1
  #run_single_glue 4 original 2
  #run_single_glue 6 original 3
  #run_single_glue 7 original 7

  #run_single_glue 3 shuffled_bigram 2
  #run_single_glue 3 shuffled_trigram 3

  # EVALUATE ANLI MODELS ON DEV-S SETS
  #run_single_glue 5 original 0 r1
  #run_single_glue 5 original 1 r2
  #run_single_glue 5 original 2 r3
  #run_single_glue 5 shuffled_trigram 0 r1
  #run_single_glue 5 shuffled_trigram 1 r2
  #run_single_glue 5 shuffled_trigram 2 r3

  # Evaluate extra-finetuned models on original (100% acc) and shuffled dev sets.
  # "roberta_synthetic/1st" is the bridge for a path to train/dev data in val_new
  #run_single_glue 1 original 0 roberta_synthetic/1st
  #run_single_glue 2 original 1 roberta_synthetic/1st
  #run_single_glue 4 original 3 roberta_synthetic/1st
  #run_single_glue 6 original 6 roberta_synthetic/1st
  #run_single_glue 7 original 7 roberta_synthetic/1st

  #run_single_glue_with_seed 1 400 original 0
  #run_single_glue_with_seed 2 400 original 1
  #run_single_glue_with_seed 4 400 original 3
  #run_single_glue_with_seed 6 400 original 6
  #run_single_glue_with_seed 7 400 original 7

  # EVALUATE SYNTHETIC MODELS on CORRESPONDING DEV SETS
  #run_single_glue 0 original 4 SST-2
  #run_single_glue 0 original 3 MRPC
  #run_single_glue 0 original 2 RTE
  #run_single_glue 0 original 1 QNLI
  #run_single_glue 0 original 0 QQP
  #run_single_glue 0 original 0 STS-B
  #run_single_glue 0 original 0 SQuAD_v2
  #run_single_glue 0 original 1 ANLI

  run_multiple_times
}

start_training 300  # 42 100 200 300 400
#start_evaluation



