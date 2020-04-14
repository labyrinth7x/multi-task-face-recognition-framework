GPUS=2,3
path='result/model/split1_labels.txt'
split=1

export CUDA_VISIBLE_DEVICES=$GPUS

python3.5 multitask.py -net ir_se -b 250 -w 3 \
    -meta_file $path \
    -pseudo_folder emore/testset \
    -remove_single \
    -device 2 \
