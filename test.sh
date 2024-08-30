# test.sh
python src/test.py \
  --task-name 'GMS_N_datav1' \
  --model 'GMS_N' \
  --n_vars 60 \
  --epochs 10 \
  --n_rounds 30 \
  --train-file 'trainset_v1_bs20000_nb3.pkl' \
  --val-file 'valset_v1_bs20000_nb1.pkl' \
  --test-file 'testset_v1_bs20000_nb75.pkl' \
  --restore 'model/GMS_N_datav1_n60_ep10_nr30_d128best_obj.pth.tar'
