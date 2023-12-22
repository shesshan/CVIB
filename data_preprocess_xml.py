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

PARSER_PATH = 'biaffine-dependency-parser-ptb-2020.04.06'
# PARSER_PATH = 'model.tar.gz'

MODELS_DIR = './parser/'
MODEL_PATH = os.path.join(MODELS_DIR, PARSER_PATH)

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--model_path',
                        type=str,
                        default=MODEL_PATH,
                        help='Path to biaffine dependency parser.')
    parser.add_argument(
        '--data_path',
        type=str,
        default='./ABSA_RGAT',
        help='Directory of data with .xml raw format.')
    parser.add_argument('--dataset',
                        type=str,
                        default='lap14',
                        choices=['rest14', 'lap14', 'mams'],
                        help='Dataset to preprocess.')
    parser.add_argument('--arts_set',
                        type=str,
                        default='ALL',
                        choices=['ALL', 'REVTGT', 'REVNON', 'ADDDIFF'],
                        help='set of ARTS, ALL/REVTGT/REVNON/ADDDIFF.')

    return parser.parse_args()

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
            f.write(s + '\n')
    print('processed', len(sent_list), 'of', file_path)


def json2txt(file_path):
    '''
    Read the original json file of semeval data and extract the text that have aspect terms.
    Store them in txt file.
    '''
    sent_list = []
    # adv1_list, adv2_list, adv3_list = [], [], []
    with open(file_path) as f:
        raw = json.load(f)  # dict
        for k, v in raw.items():
            sent = v['sentence']
            term = v['term']
            if term is None:
                continue
            if term:
                sent_list.append(sent)
    output = file_path.replace('.json', '.txt')
    with open(output, 'w') as f:
        for s in sent_list:
            f.write(s + '\n')
    print('ARTS {}: {} --> file: {}'.format(file_path.split('_')[-1].split('.')[0], len(sent_list), file_path))


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
    sentence['predicted_dependencies'] = predicted_dependencies
    sentence['predicted_heads'] = predicted_heads
    sentence['dependencies'] = []
    for idx, item in enumerate(predicted_dependencies):
        dep_tag = item
        frm = predicted_heads[idx]
        to = idx + 1
        sentence['dependencies'].append([dep_tag, frm, to])

    return sentence


def get_dependencies(file_path, predictor):
    docs = text2docs(file_path, predictor)
    sentences = [dependencies2format(doc) for doc in docs]  # list of dict
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
            example['predicted_dependencies'] = sentences[idx][
                'predicted_dependencies']
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

                left_word_offset = len(
                    tk.tokenize(example['sentence'][:left_index]))
                to_word_offset = len(
                    tk.tokenize(example['sentence'][:right_index]))

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
            example['predicted_dependencies'] = sentences[idx][
                'predicted_dependencies']
            example['predicted_heads'] = sentences[idx]['predicted_heads']
            example['dependencies'] = sentences[idx]['dependencies']
            # example['energy'] = sentences[idx]['energy']

            example['aspect_sentiment'] = []
            example['from_to'] = []  # left and right offset of the target word

            if v['polarity'] == 'conflict':
                continue
            target = v['term']
            example['aspect_sentiment'].append((target, v['polarity']))

            # index in strings, we want index in tokens
            left_index = int(v['from'])
            right_index = int(v['to'])

            left_word_offset = len(
                tk.tokenize(example['sentence'][:left_index]))
            to_word_offset = len(tk.tokenize(
                example['sentence'][:right_index]))

            example['from_to'].append((left_word_offset, to_word_offset))
            if len(example['aspect_sentiment']) == 0:
                idx += 1
                continue
            json_data.append(example)
            idx += 1
    extended_filename = origin_file.replace('.json',
                                            '_biaffine_depparsed_arts.json')
    with open(extended_filename, 'w') as f:
        json.dump(json_data, f)
    print('done', len(json_data))
    print(idx)


def main():
    args = parse_args()

    predictor = Predictor.from_path(args.model_path)

    datasets = {
        'rest14': ('Restaurants_Train_v2.xml', 'Restaurants_Test_Gold.xml'),
        'lap14': ('Laptop_Train_v2.xml', 'Laptops_Test_Gold.xml'),
        'mams': ('train.xml', 'test.xml', 'val.xml'),
    }

    data = datasets[args.dataset]

    if args.dataset == 'mams':
        args.data_path = '{}/{}'.format(args.data_path, 'mams')
        train_file, test_file, val_file = data
    else:
        args.data_path = '{}/{}'.format(args.data_path, 'semeval14')
        train_file, test_file = data
        val_file = None
    # xml -> txt
    xml2txt(os.path.join(args.data_path, train_file))
    xml2txt(os.path.join(args.data_path, test_file))

    # txt -> json
    train_sentences = get_dependencies(
        os.path.join(args.data_path, train_file.replace('.xml', '_text.txt')),
        predictor)
    test_sentences = get_dependencies(
        os.path.join(args.data_path, test_file.replace('.xml', '_text.txt')),
        predictor)

    syntaxInfo2json(train_sentences, os.path.join(args.data_path, train_file))
    syntaxInfo2json(test_sentences, os.path.join(args.data_path, test_file))

    if val_file is not None:
        xml2txt(os.path.join(args.data_path, val_file))
        val_sentences = get_dependencies(
            os.path.join(args.data_path,
                         val_file.replace('.xml', '_text.txt')), predictor)
        syntaxInfo2json(val_sentences, os.path.join(args.data_path, val_file))


def process_arts():
    ''' Process ARTS enriched test sets
    '''
    args = parse_args()

    predictor = Predictor.from_path(args.model_path)

    datasets = {
        'lap14': 'laptop_test_ARTS_{}.json'.format(args.arts_set),
        'res14': 'rest_test_ARTS_{}.json'.format(args.arts_set)
    }

    file_name = datasets[args.dataset]

    test_file_path = os.path.join(args.data_path, file_name)

    json2txt(test_file_path)  # list of sentences

    test_sentences = get_dependencies(
        os.path.join(args.data_path,
                     file_name.replace('.json', '.txt')), predictor)

    syntaxInfo2json_arts(test_sentences, test_file_path)


def split_arts_set(data_file):
    adv1, adv2, adv3 = {}, {}, {}
    with open(data_file) as f:
        raw = json.load(f)  # dict of dict
        for k, v in raw.items():
            term = v['term']
            if term is None:
                continue
            if term:
                if 'adv1' in k:
                    adv1[k]=v
                elif 'adv2' in k:
                    adv2[k]=v
                elif 'adv3' in k:
                    adv3[k]=v
    write_to_json(data_file.replace('ALL', 'REVTGT'), adv1)
    write_to_json(data_file.replace('ALL', 'REVNON'), adv2)
    write_to_json(data_file.replace('ALL', 'ADDDIFF'), adv3)

def write_to_json(data_path, data):
    json_str = json.dumps(data)
    with open(data_path, 'w') as json_file:
        json_file.write(json_str)
    json_file.close()

if __name__ == "__main__":
    
    # main() # process data with .xml format, please set the dataset name as rest14, lap14 or mams
    process_arts() # process ARTS subsets

