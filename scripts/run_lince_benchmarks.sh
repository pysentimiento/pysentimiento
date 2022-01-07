# These are all dev benchmarks -- LINCE does not provide gold labels for test data
path="evaluations/lince/ner"
models=(
    #"pysentimiento/robertuito-base-uncased"
    "pysentimiento/robertuito-base-cased"
    "pysentimiento/robertuito-base-deacc"
    #"xlm-roberta-base"
    #"bert-base-cased"
    #"bert-base-uncased"
    #"dccuchile/bert-base-spanish-wwm-uncased"
    #"dccuchile/bert-base-spanish-wwm-cased"
    #"bert-base-multilingual-uncased"
)

for base_model in ${models[@]}
do
    output_path="$path/${base_model#*/}.json"
    python bin/train.py --base_model $base_model \
        --lang es --epochs 5 --task ner \
        --benchmark --benchmark_output_path $output_path \
        --times 4
done