embeddings_path="embeddings/cc.es.300.bin"

output_path="evaluations/es/ffn_cc.json"
python bin/train.py --base_model "ffn"\
    --lang es \
    --embeddings_path $embeddings_path \
    --benchmark --benchmark_output_path $output_path \
    --overwrite

output_path="evaluations/es/rnn_cc.json"
python bin/train.py --base_model "rnn" \
    --lang es  \
    --embeddings_path $embeddings_path \
    --benchmark --benchmark_output_path $output_path \
    --overwrite

output_path="evaluations/es/birnn_cc.json"
python bin/train.py --base_model "rnn" \
    --lang es  \
    --bidirectional \
    --embeddings_path $embeddings_path \
    --benchmark --benchmark_output_path $output_path \
    --overwrite

embeddings_path="embeddings/tweet_dim_300_ws_5.bin"

output_path="evaluations/es/ffn_twitter.json"
python bin/train.py --base_model "ffn"\
    --lang es \
    --embeddings_path $embeddings_path \
    --benchmark --benchmark_output_path $output_path \
    --overwrite

output_path="evaluations/es/rnn_twitter.json"
python bin/train.py --base_model "rnn" \
    --lang es  \
    --embeddings_path $embeddings_path \
    --benchmark --benchmark_output_path $output_path \
    --overwrite

output_path="evaluations/es/birnn_twitter.json"
python bin/train.py --base_model "rnn" \
    --lang es  \
    --bidirectional \
    --embeddings_path $embeddings_path \
    --benchmark --benchmark_output_path $output_path \
    --overwrite


rm -Rf lightning_logs