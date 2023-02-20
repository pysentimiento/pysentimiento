#/bin/bash
models=(
    "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"
    "dbmdz/bert-base-italian-xxl-uncased"
    "Musixmatch/umberto-commoncrawl-cased-v1"
    "dbmdz/electra-base-italian-xxl-cased-discriminator"
)


tasks=(
    "irony"
    "sentiment"
    #"emotion"
    #"hate_speech"
)

for model in "${models[@]}"
do
    for task in "${tasks[@]}"
    do
        echo "Running benchmark for $model and $task"
        python bin/train.py --base_model $model \
            --lang it \
            --task $task \
            --benchmark --times 10
    done
done