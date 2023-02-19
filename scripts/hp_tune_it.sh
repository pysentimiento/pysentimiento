#/bin/bash
models=(
    "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"
    "dbmdz/bert-base-italian-xxl-uncased"
    "Musixmatch/umberto-commoncrawl-cased-v1"
    "dbmdz/electra-base-italian-xxl-cased-discriminator"
)

tasks=(
    #"sentiment"
    #"sentiment"
    "irony"
    #"hate_speech"
)

for model in "${models[@]}"
do

    for task in "${tasks[@]}"
    do
        echo "Running hyperparameter tuning for $model and $task"
        # Run hyperparameter tuning
        python bin/hp_tune.py --model $model \
            --lang it \
            --task $task \
            --count 40
    done
done
