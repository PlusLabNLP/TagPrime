## Base Setting
export RAW_DATA_PATH="./CoNLL2003/Full/"

export OUTPUT_PATH="./processed_data/full_conll03_roberta/"
mkdir $OUTPUT_PATH
for SPLIT in train dev test
do
    python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}.txt -o $OUTPUT_PATH/$SPLIT -b roberta-large -n $SPLIT
done

export OUTPUT_PATH="./processed_data/full_conll03_bart/"
mkdir $OUTPUT_PATH
for SPLIT in train dev test
do
    python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}.txt -o $OUTPUT_PATH/$SPLIT -b facebook/bart-large -n $SPLIT
done

export OUTPUT_PATH="./processed_data/full_conll03_t5/"
mkdir $OUTPUT_PATH
for SPLIT in train dev test
do
    python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}.txt -o $OUTPUT_PATH/$SPLIT -b t5-large -n $SPLIT
done

# ## Zero shot
# export RAW_DATA_PATH="./CoNLL2003/Full/"

# export OUTPUT_PATH="./processed_data/0shotLOC_conll03_bart/"
# mkdir $OUTPUT_PATH
# for SPLIT in train dev
# do
#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}.txt -o $OUTPUT_PATH/$SPLIT -b facebook/bart-large -n $SPLIT --valid_types MISC PER ORG
# done

# for SPLIT in test
# do
#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}.txt -o $OUTPUT_PATH/$SPLIT -b facebook/bart-large -n $SPLIT
# done

# export OUTPUT_PATH="./processed_data/0shotLOC_conll03_t5/"
# mkdir $OUTPUT_PATH
# for SPLIT in train dev 
# do
#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}.txt -o $OUTPUT_PATH/$SPLIT -b t5-large -n $SPLIT --valid_types MISC PER ORG
# done

# for SPLIT in test
# do
#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}.txt -o $OUTPUT_PATH/$SPLIT -b t5-large -n $SPLIT
# done

# ## Partial
# export RAW_DATA_PATH="./CoNLL2003/"

# export OUTPUT_PATH="./processed_data/part1_conll03_bart/"
# mkdir $OUTPUT_PATH
# for SPLIT in train dev
# do
#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}_00.txt -o $OUTPUT_PATH/${SPLIT}_00 -b facebook/bart-large -n ${SPLIT}_00 --valid_types ORG

#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}_01.txt -o $OUTPUT_PATH/${SPLIT}_01 -b facebook/bart-large -n ${SPLIT}_01 --valid_types MISC

#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}_02.txt -o $OUTPUT_PATH/${SPLIT}_02 -b facebook/bart-large -n ${SPLIT}_02 --valid_types PER

#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}_03.txt -o $OUTPUT_PATH/${SPLIT}_03 -b facebook/bart-large -n ${SPLIT}_03 --valid_types LOC

#     rm $OUTPUT_PATH/${SPLIT}_*_type.json

#     # MERGE
#     python preprocessing/combine_dataset.py -i $OUTPUT_PATH --file_prefix $SPLIT -o $OUTPUT_PATH/${SPLIT}.json
# done
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH/test.txt -o $OUTPUT_PATH/test -b facebook/bart-large -n test

# ## Partial
# export RAW_DATA_PATH="./CoNLL2003/"

# export OUTPUT_PATH="./processed_data/part3_conll03_bart/"
# mkdir $OUTPUT_PATH
# for SPLIT in train dev
# do
#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}_00.txt -o $OUTPUT_PATH/${SPLIT}_00 -b facebook/bart-large -n ${SPLIT}_00 --valid_types MISC PER LOC

#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}_01.txt -o $OUTPUT_PATH/${SPLIT}_01 -b facebook/bart-large -n ${SPLIT}_01 --valid_types PER LOC ORG

#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}_02.txt -o $OUTPUT_PATH/${SPLIT}_02 -b facebook/bart-large -n ${SPLIT}_02 --valid_types MISC LOC ORG

#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}_03.txt -o $OUTPUT_PATH/${SPLIT}_03 -b facebook/bart-large -n ${SPLIT}_03 --valid_types MISC PER ORG

#     rm $OUTPUT_PATH/${SPLIT}_*_type.json

#     # MERGE
#     python preprocessing/combine_dataset.py -i $OUTPUT_PATH --file_prefix $SPLIT -o $OUTPUT_PATH/${SPLIT}.json
# done
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH/test.txt -o $OUTPUT_PATH/test -b facebook/bart-large -n test

# #===============================================================#

# ## Partial
# export RAW_DATA_PATH="./CoNLL2003/"

# export OUTPUT_PATH="./processed_data/part1_conll03_t5/"
# mkdir $OUTPUT_PATH
# for SPLIT in train dev
# do
#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}_00.txt -o $OUTPUT_PATH/${SPLIT}_00 -b t5-large -n ${SPLIT}_00 --valid_types ORG

#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}_01.txt -o $OUTPUT_PATH/${SPLIT}_01 -b t5-large -n ${SPLIT}_01 --valid_types MISC

#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}_02.txt -o $OUTPUT_PATH/${SPLIT}_02 -b t5-large -n ${SPLIT}_02 --valid_types PER

#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}_03.txt -o $OUTPUT_PATH/${SPLIT}_03 -b t5-large -n ${SPLIT}_03 --valid_types LOC

#     rm $OUTPUT_PATH/${SPLIT}_*_type.json

#     # MERGE
#     python preprocessing/combine_dataset.py -i $OUTPUT_PATH --file_prefix $SPLIT -o $OUTPUT_PATH/${SPLIT}.json
# done
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH/test.txt -o $OUTPUT_PATH/test -b t5-large -n test

# ## Partial
# export RAW_DATA_PATH="./CoNLL2003/"

# export OUTPUT_PATH="./processed_data/part3_conll03_t5/"
# mkdir $OUTPUT_PATH
# for SPLIT in train dev
# do
#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}_00.txt -o $OUTPUT_PATH/${SPLIT}_00 -b t5-large -n ${SPLIT}_00 --valid_types MISC PER LOC

#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}_01.txt -o $OUTPUT_PATH/${SPLIT}_01 -b t5-large -n ${SPLIT}_01 --valid_types PER LOC ORG

#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}_02.txt -o $OUTPUT_PATH/${SPLIT}_02 -b t5-large -n ${SPLIT}_02 --valid_types MISC LOC ORG

#     python preprocessing/process_conll03.py -i $RAW_DATA_PATH/${SPLIT}_03.txt -o $OUTPUT_PATH/${SPLIT}_03 -b t5-large -n ${SPLIT}_03 --valid_types MISC PER ORG

#     rm $OUTPUT_PATH/${SPLIT}_*_type.json

#     # MERGE
#     python preprocessing/combine_dataset.py -i $OUTPUT_PATH --file_prefix $SPLIT -o $OUTPUT_PATH/${SPLIT}.json
# done
# python preprocessing/process_conll03.py -i $RAW_DATA_PATH/test.txt -o $OUTPUT_PATH/test -b t5-large -n test

