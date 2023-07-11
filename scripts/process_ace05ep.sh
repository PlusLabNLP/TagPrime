export ACE_PATH="../../Dataset/ace_2005_td_v7/data/"

# export OUTPUT_PATH="./processed_data/ace05ep_bart"

# mkdir $OUTPUT_PATH

# python preprocessing/process_ace05ep.py -i $ACE_PATH -o $OUTPUT_PATH -s resource/splits/ACE05-EP -b facebook/bart-large -w 1 -l english

# export OUTPUT_PATH="./processed_data/ace05ep_t5"

# mkdir $OUTPUT_PATH

# python preprocessing/process_ace05ep.py -i $ACE_PATH -o $OUTPUT_PATH -s resource/splits/ACE05-EP -b t5-base -w 1 -l english

export OUTPUT_PATH="./processed_data/ace05ep_bert"

mkdir $OUTPUT_PATH

python preprocessing/process_ace05ep.py -i $ACE_PATH -o $OUTPUT_PATH -s resource/splits/ACE05-EP -b bert-large-cased -w 1 -l english

# export OUTPUT_PATH="./processed_data/ace05ep_roberta"

# mkdir $OUTPUT_PATH

# python preprocessing/process_ace05ep.py -i $ACE_PATH -o $OUTPUT_PATH -s resource/splits/ACE05-EP -b roberta-large -w 1 -l english

# export BASE_PATH="./processed_data/"
# export SPLIT_PATH="./resource/low_resource_split/ace05ep"

# for TOKENIZER_NAME in 'bart' 't5' 'bert'
# do 
#     python preprocessing/split_dataset.py -i $BASE_PATH/ace05ep_$TOKENIZER_NAME/train.w1.oneie.json -s $SPLIT_PATH/doc_list_001 -o $BASE_PATH/ace05ep_$TOKENIZER_NAME/train.001.w1.oneie.json

#     python preprocessing/split_dataset.py -i $BASE_PATH/ace05ep_$TOKENIZER_NAME/train.w1.oneie.json -s $SPLIT_PATH/doc_list_002 -o $BASE_PATH/ace05ep_$TOKENIZER_NAME/train.002.w1.oneie.json

#     python preprocessing/split_dataset.py -i $BASE_PATH/ace05ep_$TOKENIZER_NAME/train.w1.oneie.json -s $SPLIT_PATH/doc_list_003 -o $BASE_PATH/ace05ep_$TOKENIZER_NAME/train.003.w1.oneie.json

#     python preprocessing/split_dataset.py -i $BASE_PATH/ace05ep_$TOKENIZER_NAME/train.w1.oneie.json -s $SPLIT_PATH/doc_list_005 -o $BASE_PATH/ace05ep_$TOKENIZER_NAME/train.005.w1.oneie.json

#     python preprocessing/split_dataset.py -i $BASE_PATH/ace05ep_$TOKENIZER_NAME/train.w1.oneie.json -s $SPLIT_PATH/doc_list_010 -o $BASE_PATH/ace05ep_$TOKENIZER_NAME/train.010.w1.oneie.json
        
#     python preprocessing/split_dataset.py -i $BASE_PATH/ace05ep_$TOKENIZER_NAME/train.w1.oneie.json -s $SPLIT_PATH/doc_list_020 -o $BASE_PATH/ace05ep_$TOKENIZER_NAME/train.020.w1.oneie.json

#     python preprocessing/split_dataset.py -i $BASE_PATH/ace05ep_$TOKENIZER_NAME/train.w1.oneie.json -s $SPLIT_PATH/doc_list_030 -o $BASE_PATH/ace05ep_$TOKENIZER_NAME/train.030.w1.oneie.json

#     python preprocessing/split_dataset.py -i $BASE_PATH/ace05ep_$TOKENIZER_NAME/train.w1.oneie.json -s $SPLIT_PATH/doc_list_050 -o $BASE_PATH/ace05ep_$TOKENIZER_NAME/train.050.w1.oneie.json      

#     python preprocessing/split_dataset.py -i $BASE_PATH/ace05ep_$TOKENIZER_NAME/train.w1.oneie.json -s $SPLIT_PATH/doc_list_075 -o $BASE_PATH/ace05ep_$TOKENIZER_NAME/train.075.w1.oneie.json
# done