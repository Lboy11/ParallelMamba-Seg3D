nnode=2
python \
    test1.py \
    -m \
    hydra.job.chdir=False \
    world_size=${nnode} \
    distributed=False \
    mode="eval" \
    epochs=600 \
    eval_interval=2 \
    optim=adam \
    optim.lr=8e-4 \
    batch_size=2 \
    test_batch_size=2 \
    model="SegMamba" \
    model.output="list" \
    model.feature="False" \
    model.width_ratio="0.5" \
    dataset="brats3d_acn" \
    loss="HD-MI-Dice" \
    workers=6 \
    gpu_ids="'0,1'"\
    checkname="HD-MI-SegMamba"
