# These are all dev benchmarks -- LINCE does not provide gold labels for test data
task="lince_sentiment"
path="evaluations/lince/${task}/dev"
models=(
    #"pysentimiento/robertuito-base-uncased"
    #"pysentimiento/robertuito-base-cased"
    #"pysentimiento/robertuito-base-deacc"
     "xlm-roberta-base"
     "bert-base-multilingual-uncased"
    # "bert-base-cased"
    # "bert-base-uncased"
    "dccuchile/bert-base-spanish-wwm-uncased"
    # "dccuchile/bert-base-spanish-wwm-cased"
    "vinai/bertweet-base"
)

for base_model in ${models[@]}
do
    output_path="$path/${base_model#*/}.json"
    python bin/train.py --base_model $base_model \
        --lang es --epochs 5 --task $task \
        --benchmark --benchmark_output_path $output_path \
        --times 3
done