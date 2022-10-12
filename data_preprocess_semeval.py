'''
Biaffine Dependency parser from AllenNLP
'''
import argparse
import json
import os
import re
import sys
import tarfile
import tempfile
from allennlp.predictors.predictor import Predictor
from lxml import etree
from nltk.tokenize import TreebankWordTokenizer
from tqdm import tqdm

#PARSER_PATH = 'biaffine-dependency-parser-ptb-2020.04.06.tar.gz'
PARSER_PATH = 'model.tar'

MODELS_DIR = '/data1/mschang/pretrained-models'
MODEL_PATH = os.path.join(
    MODELS_DIR, PARSER_PATH)

def xml2txt(file_path):
    '''
    Read the original xml file of semeval data and extract the text that have aspect terms.
    Store them in txt file.
    '''
    output = file_path.replace('.xml', '_text.txt')
    sent_list = []
    with open(file_path, 'rb') as f:
        raw = f.read()
        root = etree.fromstring(raw)
        for sentence in root:
            sent = sentence.find('text').text
            terms = sentence.find('aspectTerms')
            if terms is None:
                continue
            if terms:
                sent_list.append(sent)
    with open(output, 'w') as f:
        for s in sent_list:
            f.write(s+'\n')
    print('processed', len(sent_list), 'of', file_path)


def json2txt(file_path):
    '''
    Read the original json file of semeval data and extract the text that have aspect terms.
    Store them in txt file.
    '''
    output = file_path.replace('.json', '_text_arts.txt')
    sent_list = []
    with open(file_path) as f:
        raw = json.load(f)  # dict
        for _, v in raw.items():
            sent = v['sentence']
            term = v['term']
            if term is None:
                continue
            if term:
                sent_list.append(sent)
    with open(output, 'w') as f:
        for s in sent_list:
            f.write(s+'\n')
    print('processed', len(sent_list), 'of', file_path)


def text2docs(file_path, predictor):
    '''
    Annotate the sentences from extracted txt file using AllenNLP's predictor.
    '''
    with open(file_path, 'r') as f:
        sentences = f.readlines()
    docs = []
    print('Predicting dependency information...')
    for i in tqdm(range(len(sentences))):
        docs.append(predictor.predict(sentence=sentences[i]))

    return docs


def dependencies2format(doc):  # doc.sentences[i]
    '''
    Format annotation: sentence of keys
                                - tokens
                                - tags
                                - predicted_dependencies
                                - predicted_heads
                                - dependencies
    '''
    sentence = {}
    sentence['tokens'] = doc['words']
    sentence['tags'] = doc['pos']
    # sentence['energy'] = doc['energy']
    predicted_dependencies = doc['predicted_dependencies']
    predicted_heads = doc['predicted_heads']
    sentence['predicted_dependencies'] = doc['predicted_dependencies']
    sentence['predicted_heads'] = doc['predicted_heads']
    sentence['dependencies'] = []
    for idx, item in enumerate(predicted_dependencies):
        dep_tag = item
        frm = predicted_heads[idx]
        to = idx + 1
        sentence['dependencies'].append([dep_tag, frm, to])

    return sentence


def get_dependencies(file_path, predictor):
    docs = text2docs(file_path, predictor)
    sentences = [dependencies2format(doc) for doc in docs]
    return sentences


def syntaxInfo2json(sentences, origin_file):
    json_data = []
    tk = TreebankWordTokenizer()
    idx = 0
    with open(origin_file, 'rb') as fopen:
        raw = fopen.read()
        root = etree.fromstring(raw)
        for sentence in root:
            example = dict()
            example["sentence"] = sentence.find('text').text

            # for RAN
            terms = sentence.find('aspectTerms')
            if terms is None:
                continue

            example['tokens'] = sentences[idx]['tokens']
            example['tags'] = sentences[idx]['tags']
            example['predicted_dependencies'] = sentences[idx]['predicted_dependencies']
            example['predicted_heads'] = sentences[idx]['predicted_heads']
            example['dependencies'] = sentences[idx]['dependencies']
            # example['energy'] = sentences[idx]['energy']

            example["aspect_sentiment"] = []
            example['from_to'] = []  # left and right offset of the target word

            for c in terms:
                if c.attrib['polarity'] == 'conflict':
                    continue
                target = c.attrib['term']
                example["aspect_sentiment"].append(
                    (target, c.attrib['polarity']))

                # index in strings, we want index in tokens
                left_index = int(c.attrib['from'])
                right_index = int(c.attrib['to'])

                left_word_offset = len(tk.tokenize(
                    example['sentence'][:left_index]))
                to_word_offset = len(tk.tokenize(
                    example['sentence'][:right_index]))

                example['from_to'].append((left_word_offset, to_word_offset))
            if len(example['aspect_sentiment']) == 0:
                idx += 1
                continue
            json_data.append(example)
            idx += 1
    extended_filename = origin_file.replace('.xml', '_biaffine_depparsed.json')
    with open(extended_filename, 'w') as f:
        json.dump(json_data, f)
    print('done', len(json_data))
    print(idx)


def syntaxInfo2json_arts(sentences, origin_file):
    json_data = []
    tk = TreebankWordTokenizer()
    mismatch_counter = 0
    idx = 0
    with open(origin_file) as fopen:
        raw = json.load(fopen)
        for _, v in raw.items():
            example = dict()
            example['sentence'] = v['sentence']

            # for RAN
            term = v['term']
            if term is None:
                continue

            example['tokens'] = sentences[idx]['tokens']
            example['tags'] = sentences[idx]['tags']
            example['predicted_dependencies'] = sentences[idx]['predicted_dependencies']
            example['predicted_heads'] = sentences[idx]['predicted_heads']
            example['dependencies'] = sentences[idx]['dependencies']
            # example['energy'] = sentences[idx]['energy']

            example['aspect_sentiment'] = []
            example['from_to'] = []  # left and right offset of the target word

            if v['polarity'] == 'conflict':
                continue
            target = v['term']
            example['aspect_sentiment'].append(
                    (target, v['polarity']))

            # index in strings, we want index in tokens
            left_index = int(v['from'])
            right_index = int(v['to'])

            left_word_offset = len(tk.tokenize(
                    example['sentence'][:left_index]))
            to_word_offset = len(tk.tokenize(
                    example['sentence'][:right_index]))

            example['from_to'].append((left_word_offset, to_word_offset))
            if len(example['aspect_sentiment']) == 0:
                idx += 1
                continue
            json_data.append(example)
            idx += 1
    extended_filename = origin_file.replace('.json', '_biaffine_depparsed_arts.json')
    with open(extended_filename, 'w') as f:
        json.dump(json_data, f)
    print('done', len(json_data))
    print(idx)


def main():
    args = parse_args()

    predictor = Predictor.from_path(args.model_path)

    datasets = {'res14': ('Restaurants_Train_v2.xml', 'Restaurants_Test_Gold.xml'),
                'lap14': ('Laptop_Train_v2.xml', 'Laptops_Test_Gold.xml'),
                'mams': ('train.xml', 'test.xml', 'val.xml'), }

    data = datasets[args.dataset]

    if args.dataset == 'mams':
        train_file, test_file, val_file = data
    else:
        train_file, test_file = data
        val_file = None
    # xml -> txt
    xml2txt(os.path.join(args.data_path, train_file))
    xml2txt(os.path.join(args.data_path, test_file))

    # txt -> json
    train_sentences = get_dependencies(
        os.path.join(args.data_path, train_file.replace('.xml', '_text.txt')), predictor)
    test_sentences = get_dependencies(os.path.join(
        args.data_path, test_file.replace('.xml', '_text.txt')), predictor)

    syntaxInfo2json(train_sentences, os.path.join(
        args.data_path, train_file))
    syntaxInfo2json(test_sentences, os.path.join(
        args.data_path, test_file))

    if val_file is not None:
        xml2txt(os.path.join(args.data_path, val_file))
        val_sentences = get_dependencies(os.path.join(
            args.data_path, val_file.replace('.xml', '_text.txt')), predictor)
        syntaxInfo2json(val_sentences, os.path.join(
            args.data_path, val_file))


def process_arts():
    ''' Process ARTS enriched test sets
    '''
    args = parse_args()

    predictor = Predictor.from_path(args.model_path)

    datasets = {'lap14': 'laptop_test_enriched.json',
                'res14': 'rest_test_enriched.json'}

    file_name = datasets[args.dataset]

    test_file_path = os.path.join(args.data_path, file_name)

    json2txt(test_file_path)

    test_sentences = get_dependencies(os.path.join(
        args.data_path, file_name.replace('.json', '_text_arts.txt')), predictor)

    syntaxInfo2json_arts(test_sentences, test_file_path)

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                        help='Path to biaffine dependency parser.')
    parser.add_argument('--data_path', type=str, default='/data1/mschang/ABSA/data/semeval14',
                        help='Directory of where SemEval or twiiter or MAMS data held.')
    parser.add_argument('--dataset', type=str, default='lap14',
                        help='Dataset to preprocess.')

    return parser.parse_args()


if __name__ == "__main__":
    '''
    tmp_path='./data/pretrained-models/tmp'
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    tempdir = os.path.join('./data/pretrained-models','tmp')
    with tarfile.open(MODEL_PATH, "r:gz") as archive:
            archive.extractall(tempdir)
    '''
    
    #main()
    process_arts()
