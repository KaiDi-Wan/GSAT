# generate_data.sh
# Should be executed after the raw data and solution have been prepared.
#
# NOTE:
# Format of 'data_selector': [[clen, nvar, nclauses, id_begin, id_end], ...]
#

python src/generate_data.py \
    --raw_data_path "raw_data_uf" \
    --data_selector "[[2, 60, 600, 1, 8000]]" \
    --data_type "train" \
    --prefix "trainset_v1" \
    --batch_size "10000"

