import nltk
from nltk.stem import WordNetLemmatizer
import json
import spacy
from tqdm import tqdm
import warnings
import argparse
import os
warnings.filterwarnings("ignore", category=UserWarning)

# 延迟加载 spaCy 模型
_nlp = None

def get_nlp():
    """延迟加载 spaCy 模型"""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_lg")
        except OSError:
            raise OSError("spaCy 模型 'en_core_web_lg' 未安装。请运行: python -m spacy download en_core_web_lg")
    return _nlp


inference_data_path = ""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_association", type=str, default='.../AMBER/data/relation.json')
    parser.add_argument("--safe_words", type=str, default='.../AMBER/data/safe_words.txt')
    parser.add_argument("--inference_data", type=str, default=inference_data_path)
    parser.add_argument("--annotation", type=str, default='.../AMBER/data/annotations.json')
    parser.add_argument("--metrics", type=str, default='.../AMBER/data/metrics.txt')
    parser.add_argument("--similarity_score", type=float, default=0.8)
    parser.add_argument('--evaluation_type', choices=['a', 'g', 'd', 'de', 'da', 'dr'], default='g')
    # help='a: all tasks and dimensions    g: generative task    d: descriminative task    de, da, dr: existence, attribute, relation'
    args = parser.parse_args()
    return args


def check_synonyms_word(word1, word2, similarity_score, nlp=None):
    """检查两个词是否为同义词"""
    if nlp is None:
        nlp = get_nlp()
    token1 = nlp(word1)
    token2 = nlp(word2)
    similarity = token1.similarity(token2)
    return similarity > similarity_score


def extract_nouns(text):
    """从文本中提取名词"""
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    nouns = [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith('NN')]
    return nouns


def init_metrics(metrics_file):
    """初始化指标字典"""
    metrics = {}
    with open(metrics_file, "r") as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split('=')
        if len(parts) == 2:
            variable_name = parts[0].strip()
            variable_value = eval(parts[1].strip())
            metrics[variable_name] = variable_value

    return metrics


def evaluate_amber_core(inference_data, ground_truth, metrics, association, global_safe_words,
                        similarity_score, dimension, nlp=None, verbose=True):
    """
    核心评估函数，处理推理数据并更新指标

    Args:
        inference_data: 推理结果列表（JSON 格式）
        ground_truth: 标注数据（JSON 格式）
        metrics: 指标字典（会被修改）
        association: 关联词字典
        global_safe_words: 全局安全词列表
        similarity_score: 相似度阈值
        dimension: 评估维度字典
        nlp: spaCy 模型（可选，如果为 None 则自动加载）
        verbose: 是否显示进度条

    Returns:
        dict: 包含计算结果的字典
    """
    if nlp is None:
        nlp = get_nlp()

    hallucination_words = []
    for word1 in association.keys():
        hallucination_words.append(word1)
        for word2 in association.get(word1, []):
            hallucination_words.append(word2)

    iterator = tqdm(range(len(inference_data)), desc="评估进度") if verbose else range(len(inference_data))

    for i in iterator:
        id = inference_data[i]['id']

        if ground_truth[id-1]['type'] == 'generative':
            nouns = extract_nouns(inference_data[i]['response'])
            after_process_nouns = []
            for noun in nouns:
                if noun in hallucination_words:
                    after_process_nouns.append(noun)

            safe_words = []
            safe_list = []
            for idx, word in enumerate(ground_truth[id-1]['truth']):
                safe_words += association.get(word, [])
                safe_list += [idx] * len(association.get(word, []))

            ha_words = []
            ha_list = []
            for idx, word in enumerate(ground_truth[id-1]['hallu']):
                ha_words += association.get(word, [])
                ha_list += [idx] * len(association.get(word, []))

            safe_words += ground_truth[id-1]['truth']
            safe_len = len(ground_truth[id-1]['truth'])
            safe_list += [0] * safe_len
            safe_flag_list = [0] * len(after_process_nouns)

            ha_words += ground_truth[id-1]['hallu']
            ha_len = len(ground_truth[id-1]['hallu'])
            ha_list += [0] * ha_len

            for idx, noun in enumerate(after_process_nouns):
                if noun in global_safe_words:
                    continue

                if noun in safe_words:
                    for j in range(len(safe_words)):
                        if noun == safe_words[j]:
                            if j < (len(safe_list) - safe_len):
                                safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                            else:
                                safe_list[j] = 1
                            break
                    continue

                if noun in ha_words:
                    for j in range(len(ha_words)):
                        if noun == ha_words[j]:
                            if j < (len(ha_list) - ha_len):
                                ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                            else:
                                ha_list[j] = 1
                            break

                for j, check_word in enumerate(ha_words):
                    if check_synonyms_word(noun, check_word, similarity_score, nlp):
                        if j < (len(ha_list) - ha_len):
                            ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                        else:
                            ha_list[j] = 1
                        break

                flag = False
                for j, check_word in enumerate(safe_words):
                    if check_synonyms_word(noun, check_word, similarity_score, nlp):
                        flag = True
                        if j < (len(safe_list) - safe_len):
                            safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                        else:
                            safe_list[j] = 1
                        break
                if flag == True:
                    continue

                safe_flag_list[idx] = 1

            metrics['chair_score'] += sum(safe_flag_list)
            metrics['chair_num'] += len(safe_flag_list)
            metrics['safe_cover_score'] += sum(safe_list[-safe_len:])
            metrics['safe_cover_num'] += len(safe_list[-safe_len:])
            metrics['hallu_cover_score'] += sum(ha_list[-ha_len:])
            metrics['hallu_cover_num'] += len(ha_list[-ha_len:])
            if sum(safe_flag_list) == 0:
                metrics['non_hallu_score'] += 1
            metrics['non_hallu_num'] += 1

        else:
            metrics['qa_correct_num'] += 1
            if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                metrics['as_qa_correct_num'] += 1
            elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                metrics['an_qa_correct_num'] += 1
            elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                metrics['aa_qa_correct_num'] += 1
            elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                metrics['ha_qa_correct_num'] += 1
            else:
                metrics['asso_qa_correct_num'] += 1

            truth = ground_truth[id-1]['truth']
            response = inference_data[i]['response'].strip()
            # 标准化响应格式（处理大小写和空格）
            response_normalized = response.lower().strip()
            if truth == 'yes':
                if response_normalized == 'yes' or response == 'Yes':
                    metrics['qa_correct_score'] += 1
                    if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                        metrics['as_qa_correct_score'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                        metrics['an_qa_correct_score'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                        metrics['aa_qa_correct_score'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                        metrics['ha_qa_correct_score'] += 1
                    else:
                        metrics['asso_qa_correct_score'] += 1
            else:
                metrics['qa_no_num'] += 1
                if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                    metrics['as_qa_no_num'] += 1
                elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                    metrics['an_qa_no_num'] += 1
                elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                    metrics['aa_qa_no_num'] += 1
                elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                    metrics['ha_qa_no_num'] += 1
                else:
                    metrics['asso_qa_no_num'] += 1

                if response_normalized == 'no' or response == 'No':
                    metrics['qa_correct_score'] += 1
                    metrics['qa_no_score'] += 1
                    if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                        metrics['as_qa_correct_score'] += 1
                        metrics['as_qa_no_score'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                        metrics['an_qa_correct_score'] += 1
                        metrics['an_qa_no_score'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                        metrics['aa_qa_correct_score'] += 1
                        metrics['aa_qa_no_score'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                        metrics['ha_qa_correct_score'] += 1
                        metrics['ha_qa_no_score'] += 1
                    else:
                        metrics['asso_qa_correct_score'] += 1
                        metrics['asso_qa_no_score'] += 1

            if response_normalized == 'no' or response == 'No':
                metrics['qa_ans_no_num'] += 1
                if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                    metrics['as_qa_ans_no_num'] += 1
                elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                    metrics['an_qa_ans_no_num'] += 1
                elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                    metrics['aa_qa_ans_no_num'] += 1
                elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                    metrics['ha_qa_ans_no_num'] += 1
                else:
                    metrics['asso_qa_ans_no_num'] += 1
                if truth == 'no':
                    metrics['qa_ans_no_score'] += 1
                    if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                        metrics['as_qa_ans_no_score'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                        metrics['an_qa_ans_no_score'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                        metrics['aa_qa_ans_no_score'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                        metrics['ha_qa_ans_no_score'] += 1
                    else:
                        metrics['asso_qa_ans_no_score'] += 1

    # 计算最终指标
    results = {}

    if dimension['g']:
        CHAIR = round(metrics['chair_score'] / metrics['chair_num'] * 100, 1) if metrics['chair_num'] > 0 else 0.0
        Cover = round(metrics['safe_cover_score'] / metrics['safe_cover_num'] * 100, 1) if metrics['safe_cover_num'] > 0 else 0.0
        Ha = round(metrics['hallu_cover_score'] / metrics['hallu_cover_num'] * 100, 1) if metrics['hallu_cover_num'] > 0 else 0.0
        Ha_p = round(100 - metrics['non_hallu_score'] / metrics['non_hallu_num'] * 100, 1) if metrics['non_hallu_num'] > 0 else 0.0
        results['generative'] = {
            'CHAIR': CHAIR,
            'Cover': Cover,
            'Hal': Ha_p,
            'Cog': Ha
        }

    if dimension['de'] and dimension['da'] and dimension['dr']:
        Accuracy = round(metrics['qa_correct_score'] / metrics['qa_correct_num'] * 100, 1) if metrics['qa_correct_num'] > 0 else 0.0
        Precision = round(metrics['qa_ans_no_score'] / metrics['qa_ans_no_num'] * 100, 1) if metrics['qa_ans_no_num'] > 0 else 0.0
        Recall = round(metrics['qa_no_score'] / metrics['qa_no_num'] * 100, 1) if metrics['qa_no_num'] > 0 else 0.0
        F1 = round(2 * (Precision/100) * (Recall/100) / ((Precision/100) + (Recall/100) + 0.0001) * 100, 1) if (Precision + Recall) > 0 else 0.0
        results['discriminative'] = {
            'Accuracy': Accuracy,
            'Precision': Precision,
            'Recall': Recall,
            'F1': F1
        }

    return results


def evaluate_amber_from_files(answers_file, amber_data_path, similarity_score=0.8, evaluation_type='a',
                               answers_format='jsonl', verbose=True):
    """
    从文件评估 AMBER 答案（支持 JSONL 和 JSON 格式）

    Args:
        answers_file: 答案文件路径（JSONL 或 JSON 格式）
        amber_data_path: AMBER 数据根目录
        similarity_score: 相似度阈值
        evaluation_type: 评估类型 ('a': all, 'g': generative, 'd': discriminative, 'de': existence, 'da': attribute, 'dr': relation)
        answers_format: 答案文件格式 ('jsonl' 或 'json')
        verbose: 是否显示进度条

    Returns:
        tuple: (results_dict, metrics_dict)
    """
    # 检查并下载 NLTK 数据
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("正在下载 NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("正在下载 NLTK POS tagger...")
        nltk.download('averaged_perceptron_tagger', quiet=True)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("正在下载 NLTK WordNet...")
        nltk.download('wordnet', quiet=True)

    # 加载 spaCy 模型
    nlp = get_nlp()

    # 初始化指标
    metrics_file = os.path.join(amber_data_path, "metrics.txt")
    metrics = init_metrics(metrics_file)

    # 加载关联词和安全词
    relation_file = os.path.join(amber_data_path, "relation.json")
    safe_words_file = os.path.join(amber_data_path, "safe_words.txt")
    annotation_file = os.path.join(amber_data_path, "annotations.json")

    association = json.load(open(relation_file, 'r', encoding='utf-8'))

    global_safe_words = []
    with open(safe_words_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('\n')[0]
            global_safe_words.append(line)

    # 加载答案
    if answers_format == 'jsonl':
        inference_data = []
        with open(answers_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    inference_data.append(json.loads(line.strip()))
    else:
        inference_data = json.load(open(answers_file, 'r', encoding='utf-8'))

    # 加载标注
    ground_truth = json.load(open(annotation_file, 'r', encoding='utf-8'))

    # 确定评估维度
    dimension = {'g': False, 'de': False, 'da': False, 'dr': False}
    if evaluation_type == 'a':
        for key in dimension.keys():
            dimension[key] = True
    elif evaluation_type == 'g':
        dimension['g'] = True
    elif evaluation_type == 'd':
        dimension['de'] = True
        dimension['da'] = True
        dimension['dr'] = True
    else:
        dimension[evaluation_type] = True

    # 执行评估
    results = evaluate_amber_core(
        inference_data=inference_data,
        ground_truth=ground_truth,
        metrics=metrics,
        association=association,
        global_safe_words=global_safe_words,
        similarity_score=similarity_score,
        dimension=dimension,
        nlp=nlp,
        verbose=verbose
    )

    return results, metrics


def main(args):
    """命令行入口函数"""
    metrics = init_metrics(args.metrics)
    association = json.load(open(args.word_association, 'r', encoding='utf-8'))

    global_safe_words = []
    with open(args.safe_words, 'r', encoding='utf-8') as safe_file:
        for line in safe_file:
            line = line.split('\n')[0]
            global_safe_words.append(line)

    dimension = {'g': False,'de': False, 'da': False, 'dr': False}
    if args.evaluation_type == 'a':
        for key in dimension.keys():
            dimension[key] = True
    elif args.evaluation_type == 'g':
        dimension['g'] = True
    elif args.evaluation_type == 'd':
        dimension['de'] = True
        dimension['da'] = True
        dimension['dr'] = True
    else:
        dimension[args.evaluation_type] = True

    inference_data = json.load(open(args.inference_data, 'r', encoding='utf-8'))
    ground_truth = json.load(open(args.annotation, 'r', encoding='utf-8'))

    # 使用重构后的核心函数
    nlp = get_nlp()
    results = evaluate_amber_core(
        inference_data=inference_data,
        ground_truth=ground_truth,
        metrics=metrics,
        association=association,
        global_safe_words=global_safe_words,
        similarity_score=args.similarity_score,
        dimension=dimension,
        nlp=nlp,
        verbose=True
    )

    # 为了兼容原有的打印逻辑，需要计算额外的指标
    # 这些指标在 evaluate_amber_core 中已经计算，但为了保持输出格式，我们需要重新计算

    # 打印结果（保持原有输出格式）
    if dimension['g']:
        print("Generative Task:")
        print("CHAIR:\t\t", results['generative']['CHAIR'])
        print("Cover:\t\t", results['generative']['Cover'])
        print("Hal:\t\t", results['generative']['Hal'])
        print("Cog:\t\t", results['generative']['Cog'], "\n")

    if dimension['de'] and dimension['da'] and dimension['dr']:
        print("Descriminative Task:")
        print("Accuracy:\t", results['discriminative']['Accuracy'])
        print("Precision:\t", results['discriminative']['Precision'])
        print("Recall:\t\t", results['discriminative']['Recall'])
        print("F1:\t\t", results['discriminative']['F1'], "\n")

    if dimension['de']:
        hallucination_Accuracy = round(metrics['ha_qa_correct_score'] / metrics['ha_qa_correct_num'] * 100, 1) if metrics['ha_qa_correct_num'] > 0 else 0.0
        hallucination_Precision = round(metrics['ha_qa_ans_no_score'] / metrics['ha_qa_ans_no_num'] * 100, 1) if metrics['ha_qa_ans_no_num'] > 0 else 0.0
        hallucination_Recall = round(metrics['ha_qa_no_score'] / metrics['ha_qa_no_num'] * 100, 1) if metrics['ha_qa_no_num'] > 0 else 0.0
        hallucination_F1 = round(2 * (hallucination_Precision/100) * (hallucination_Recall/100) / ((hallucination_Precision/100) + (hallucination_Recall/100) + 0.001) * 100, 1) if (hallucination_Precision + hallucination_Recall) > 0 else 0.0
        print("Exsitence:")
        print("Accuracy:\t", hallucination_Accuracy)
        print("Precision:\t", hallucination_Precision)
        print("Recall:\t\t", hallucination_Recall)
        print("F1:\t\t", hallucination_F1, "\n")

    if dimension['da']:
        attr_Accuracy = round((metrics['as_qa_correct_score'] + metrics['an_qa_correct_score'] + metrics['aa_qa_correct_score']) / (metrics['as_qa_correct_num'] + metrics['an_qa_correct_num'] + metrics['aa_qa_correct_num']) * 100, 1) if (metrics['as_qa_correct_num'] + metrics['an_qa_correct_num'] + metrics['aa_qa_correct_num']) > 0 else 0.0
        attr_Precision = round((metrics['as_qa_ans_no_score'] + metrics['an_qa_ans_no_score'] + metrics['aa_qa_ans_no_score']) / (metrics['as_qa_ans_no_num'] + metrics['an_qa_ans_no_num'] + metrics['aa_qa_ans_no_num']) * 100, 1) if (metrics['as_qa_ans_no_num'] + metrics['an_qa_ans_no_num'] + metrics['aa_qa_ans_no_num']) > 0 else 0.0
        attr_Recall = round((metrics['as_qa_no_score'] + metrics['an_qa_no_score'] + metrics['aa_qa_no_score']) / (metrics['as_qa_no_num'] + metrics['an_qa_no_num'] + metrics['aa_qa_no_num']) * 100, 1) if (metrics['as_qa_no_num'] + metrics['an_qa_no_num'] + metrics['aa_qa_no_num']) > 0 else 0.0
        attr_F1 = round(2 * (attr_Precision/100) * (attr_Recall/100) / ((attr_Precision/100) + (attr_Recall/100) + 0.0001) * 100, 1) if (attr_Precision + attr_Recall) > 0 else 0.0
        state_Accuracy = round(metrics['as_qa_correct_score'] / metrics['as_qa_correct_num'] * 100, 1) if metrics['as_qa_correct_num'] > 0 else 0.0
        state_Precision = round(metrics['as_qa_ans_no_score'] / metrics['as_qa_ans_no_num'] * 100, 1) if metrics['as_qa_ans_no_num'] > 0 else 0.0
        state_Recall = round(metrics['as_qa_no_score'] / metrics['as_qa_no_num'] * 100, 1) if metrics['as_qa_no_num'] > 0 else 0.0
        state_F1 = round(2 * (state_Precision/100) * (state_Recall/100) / ((state_Precision/100) + (state_Recall/100) + 0.0001) * 100, 1) if (state_Precision + state_Recall) > 0 else 0.0
        number_Accuracy = round(metrics['an_qa_correct_score'] / metrics['an_qa_correct_num'] * 100, 1) if metrics['an_qa_correct_num'] > 0 else 0.0
        number_Precision = round(metrics['an_qa_ans_no_score'] / metrics['an_qa_ans_no_num'] * 100, 1) if metrics['an_qa_ans_no_num'] > 0 else 0.0
        number_Recall = round(metrics['an_qa_no_score'] / metrics['an_qa_no_num'] * 100, 1) if metrics['an_qa_no_num'] > 0 else 0.0
        number_F1 = round(2 * (number_Precision/100) * (number_Recall/100) / ((number_Precision/100) + (number_Recall/100) + 0.0001) * 100, 1) if (number_Precision + number_Recall) > 0 else 0.0
        action_Accuracy = round(metrics['aa_qa_correct_score'] / metrics['aa_qa_correct_num'] * 100, 1) if metrics['aa_qa_correct_num'] > 0 else 0.0
        action_Precision = round(metrics['aa_qa_ans_no_score'] / metrics['aa_qa_ans_no_num'] * 100, 1) if metrics['aa_qa_ans_no_num'] > 0 else 0.0
        action_Recall = round(metrics['aa_qa_no_score'] / metrics['aa_qa_no_num'] * 100, 1) if metrics['aa_qa_no_num'] > 0 else 0.0
        action_F1 = round(2 * (action_Precision/100) * (action_Recall/100) / ((action_Precision/100) + (action_Recall/100) + 0.0001) * 100, 1) if (action_Precision + action_Recall) > 0 else 0.0
        print("Attribute:")
        print("Accuracy:\t", attr_Accuracy)
        print("Precision:\t", attr_Precision)
        print("Recall:\t\t", attr_Recall)
        print("F1:\t\t", attr_F1, "\n")
        print("State:")
        print("Accuracy:\t", state_Accuracy)
        print("Precision:\t", state_Precision)
        print("Recall:\t\t", state_Recall)
        print("F1:\t\t", state_F1, "\n")
        print("Number:")
        print("Accuracy:\t", number_Accuracy)
        print("Precision:\t", number_Precision)
        print("Recall:\t\t", number_Recall)
        print("F1:\t\t", number_F1, "\n")
        print("Action:")
        print("Accuracy:\t", action_Accuracy)
        print("Precision:\t", action_Precision)
        print("Recall:\t\t", action_Recall)
        print("F1:\t\t", action_F1, "\n")

    if dimension['dr']:
        relation_Accuracy = round(metrics['asso_qa_correct_score'] / metrics['asso_qa_correct_num'] * 100, 1) if metrics['asso_qa_correct_num'] > 0 else 0.0
        relation_Precision = round(metrics['asso_qa_ans_no_score'] / metrics['asso_qa_ans_no_num'] * 100, 1) if metrics['asso_qa_ans_no_num'] > 0 else 0.0
        relation_Recall = round(metrics['asso_qa_no_score'] / metrics['asso_qa_no_num'] * 100, 1) if metrics['asso_qa_no_num'] > 0 else 0.0
        relation_F1 = round(2 * (relation_Precision/100) * (relation_Recall/100) / ((relation_Precision/100) + (relation_Recall/100) + 0.0001) * 100, 1) if (relation_Precision + relation_Recall) > 0 else 0.0
        print("Relation:")
        print("Accuracy:\t", relation_Accuracy)
        print("Precision:\t", relation_Precision)
        print("Recall:\t\t", relation_Recall)
        print("F1:\t\t", relation_F1)

if __name__ == "__main__":
    args = get_args()
    main(args)