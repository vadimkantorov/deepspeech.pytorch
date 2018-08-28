#!/bin/dash
#files="data/models/en/ted/deepspeech_?.pth data/models/en/ted/deepspeech_??.pth"
files="data/models/en/ted/deepspeech_*.pth"
fn=/tmp/.cp.val
bs=56
echo >$fn.last
#ls $files >$fn.last
val () {
    i=$1
    t=$2
    mf=$3
    if [ ! -s $i.$t ]; then
        echo "Processing $i for $t"
        CUDA_VISIBLE_DEVICES=1 .venv/bin/python test.py --test-manifest $mf --batch-size $bs --cuda --decoder beam --model-path $i | tee -a $i.$t
    fi
    echo -n $i "\t" >>data/logs/$t.log
    cat $i.$t >>data/logs/$t.log
}

echo '' >data/logs/val.log
echo '' >data/logs/clean.log
echo '' >data/logs/ted.log
while [ true ]; do
    ls $files >$fn.new
    new=$(comm -23 $fn.new $fn.last)
    sleep 1;
    if [ "x" = "x$new" ]; then continue; fi
    cp $fn.new $fn.last
    for f in $new; do
        val $f "clean" "data/manifests/en/merged_test_clean.csv";
        val $f "ted" "data/manifests/en/merged_ted.csv";
        val $f "val" "data/manifests/en/merged_val_manifest.csv";
    done
    echo 'updated logs'
done
