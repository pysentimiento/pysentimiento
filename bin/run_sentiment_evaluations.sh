embeddings_path="embeddings/cc.es.300.bin"
output_path="evaluations/sentiment/es/rnn_cc.json"
model_name="rnn"
python bin/train.py --base_model $model_name\
    --lang es --task sentiment \
    --embeddings_path $embeddings_path \
    --benchmark --benchmark_output_path $output_path

embeddings_path="embeddings/tweet_dim_300_ws_5.bin"
output_path="evaluations/sentiment/es/rnn_twitter.json"
model_name="rnn"
python bin/train.py --base_model $model_name\
    --lang es --task sentiment \
    --embeddings_path $embeddings_path \
    --benchmark --benchmark_output_path $output_path
