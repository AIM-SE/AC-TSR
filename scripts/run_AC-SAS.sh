DATASET=$1

python run_recbole.py \
  --model=ACSASRec \
  --dataset=$DATASET \
  --config_files="config/${DATASET}.yaml config/config_t.yaml"
