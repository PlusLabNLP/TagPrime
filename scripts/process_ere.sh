export ERE_PATH="../../../Dataset/ERE_EN/"

# export OUTPUT_PATH="./processed_data/ere_bart"

# mkdir $OUTPUT_PATH

# python preprocessing/process_ere.py -i $ERE_PATH -o $OUTPUT_PATH -s resource/splits/ERE-EN -b facebook/bart-large -w 1

# export OUTPUT_PATH="./processed_data/ere_t5"

# mkdir $OUTPUT_PATH

# python preprocessing/process_ere.py -i $ERE_PATH -o $OUTPUT_PATH -s resource/splits/ERE-EN -b t5-base -w 1

export OUTPUT_PATH="./processed_data/ere_bert"

mkdir $OUTPUT_PATH

python preprocessing/process_ere.py -i $ERE_PATH -o $OUTPUT_PATH -s resource/splits/ERE-EN -b bert-large-cased -w 1

# export OUTPUT_PATH="./processed_data/ere_roberta"

# mkdir $OUTPUT_PATH

# python preprocessing/process_ere.py -i $ERE_PATH -o $OUTPUT_PATH -s resource/splits/ERE-EN -b roberta-large -w 1


# export BASE_PATH="./processed_data/"
# export SPLIT_PATH="./resource/low_resource_split/ere"

# for TOKENIZER_NAME in 'bart' 't5' 'bert'
# do 
#     python preprocessing/split_dataset.py -i $BASE_PATH/ere_$TOKENIZER_NAME/train.w1.oneie.json -s $SPLIT_PATH/doc_list_001 -o $BASE_PATH/ere_$TOKENIZER_NAME/train.001.w1.oneie.json

#     python preprocessing/split_dataset.py -i $BASE_PATH/ere_$TOKENIZER_NAME/train.w1.oneie.json -s $SPLIT_PATH/doc_list_002 -o $BASE_PATH/ere_$TOKENIZER_NAME/train.002.w1.oneie.json

#     python preprocessing/split_dataset.py -i $BASE_PATH/ere_$TOKENIZER_NAME/train.w1.oneie.json -s $SPLIT_PATH/doc_list_003 -o $BASE_PATH/ere_$TOKENIZER_NAME/train.003.w1.oneie.json

#     python preprocessing/split_dataset.py -i $BASE_PATH/ere_$TOKENIZER_NAME/train.w1.oneie.json -s $SPLIT_PATH/doc_list_005 -o $BASE_PATH/ere_$TOKENIZER_NAME/train.005.w1.oneie.json

#     python preprocessing/split_dataset.py -i $BASE_PATH/ere_$TOKENIZER_NAME/train.w1.oneie.json -s $SPLIT_PATH/doc_list_010 -o $BASE_PATH/ere_$TOKENIZER_NAME/train.010.w1.oneie.json
        
#     python preprocessing/split_dataset.py -i $BASE_PATH/ere_$TOKENIZER_NAME/train.w1.oneie.json -s $SPLIT_PATH/doc_list_020 -o $BASE_PATH/ere_$TOKENIZER_NAME/train.020.w1.oneie.json

#     python preprocessing/split_dataset.py -i $BASE_PATH/ere_$TOKENIZER_NAME/train.w1.oneie.json -s $SPLIT_PATH/doc_list_030 -o $BASE_PATH/ere_$TOKENIZER_NAME/train.030.w1.oneie.json

#     python preprocessing/split_dataset.py -i $BASE_PATH/ere_$TOKENIZER_NAME/train.w1.oneie.json -s $SPLIT_PATH/doc_list_050 -o $BASE_PATH/ere_$TOKENIZER_NAME/train.050.w1.oneie.json      

#     python preprocessing/split_dataset.py -i $BASE_PATH/ere_$TOKENIZER_NAME/train.w1.oneie.json -s $SPLIT_PATH/doc_list_075 -o $BASE_PATH/ere_$TOKENIZER_NAME/train.075.w1.oneie.json
# done