'''
Copied from: https://github.com/LisaAnne/Hallucination/blob/master/utils/chair.py

Modified by: Maxlinn

1. adapt calculation of CHAIR-i and CHAIR-s for Python3, supports for both json and jsonl file input.
2. integrate synonyms.txt to make the script standalone.
3. remove machine-translation based metrics BLEU-n, CIDEr, ROGUE
4. add new metric Recall, which represents the node words(i.e. lemmas of objects) coverage overall.
5. add pickle cache mechanism to make it fast for repetitive evaluations.
'''


import os
import sys
import json
# from pattern.en import singularize
import argparse
import tqdm
import pickle
from collections import defaultdict

# 延迟导入 nltk，并提供友好的错误提示
try:
    import nltk
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer

    # 检查并下载必要的 nltk 数据
    # 新版本的 NLTK (3.8.1+) 使用 punkt_tab，旧版本使用 punkt
    punkt_available = False
    try:
        nltk.data.find('tokenizers/punkt_tab')
        punkt_available = True
    except LookupError:
        try:
            nltk.data.find('tokenizers/punkt')
            punkt_available = True
        except LookupError:
            pass

    if not punkt_available:
        print("正在下载 NLTK punkt tokenizer 数据...")
        # 优先尝试下载新版本的 punkt_tab
        try:
            nltk.download('punkt_tab', quiet=True)
            print("✓ 已下载 punkt_tab")
        except Exception as e:
            # 如果失败，尝试旧版本的 punkt
            try:
                nltk.download('punkt', quiet=True)
                print("✓ 已下载 punkt (旧版本)")
            except Exception as e2:
                print(f"⚠️  警告: 下载 punkt 数据失败: {e2}")
                raise

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("正在下载 NLTK POS tagger 数据...")
        nltk.download('averaged_perceptron_tagger', quiet=True)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("正在下载 NLTK wordnet 数据...")
        nltk.download('wordnet', quiet=True)

except ImportError as e:
    print("=" * 80)
    print("错误: 缺少 NLTK 库")
    print("=" * 80)
    print("请安装 NLTK:")
    print("  pip install nltk")
    print("")
    print("安装后，还需要下载 NLTK 数据:")
    print("  python -c \"import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')\"")
    print("  或者（旧版本 NLTK < 3.8.1）:")
    print("  python -c \"import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')\"")
    print("=" * 80)
    raise


# copied from: https://github.com/LisaAnne/Hallucination/blob/master/data/synonyms.txt
synonyms_txt = '''
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter,  motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake,  cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
'''


def combine_coco_captions(annotation_path):

    if not os.path.exists('%s/captions_%s2014.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO caption annotations for val set")
    if not os.path.exists('%s/captions_%s2014.json' %(annotation_path, 'train')):
        raise Exception("Please download MSCOCO caption annotations for train set")

    val_caps = json.load(open('%s/captions_%s2014.json' %(annotation_path, 'val')))
    train_caps = json.load(open('%s/captions_%s2014.json' %(annotation_path, 'train')))
    all_caps = {'info': train_caps['info'],
                'licenses': train_caps['licenses'],
                'images': val_caps['images'] + train_caps['images'],
                'annotations': val_caps['annotations'] + train_caps['annotations']}

    return all_caps

def combine_coco_instances(annotation_path):

    if not os.path.exists('%s/instances_%s2014.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO instance annotations for val set")
    if not os.path.exists('%s/instances_%s2014.json' %(annotation_path, 'train')):
        raise Exception("Please download MSCOCO instance annotations for train set")

    val_instances = json.load(open('%s/instances_%s2014.json' %(annotation_path, 'val')))
    train_instances = json.load(open('%s/instances_%s2014.json' %(annotation_path, 'train')))
    all_instances = {'info': train_instances['info'],
                     'licenses': train_instances['licenses'],
                     'type': train_instances['licenses'],
                     'categories': train_instances['categories'],
                     'images': train_instances['images'] + val_instances['images'],
                     'annotations': val_instances['annotations'] + train_instances['annotations']}

    return all_instances

class CHAIR(object):

    def __init__(self, coco_path):

        self.imid_to_objects = defaultdict(list) # later become a dict of sets

        self.coco_path = coco_path

        #read in synonyms
        synonyms = synonyms_txt.splitlines()
        synonyms = [s.strip().split(', ') for s in synonyms]
        self.mscoco_objects = [] #mscoco objects and *all* synonyms
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        #Some hard coded rules for implementing CHAIR metrics on MSCOCO

        #common 'double words' in MSCOCO that should be treated as a single word
        coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track']

        #Hard code some rules for special cases in MSCOCO
        #qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
        #qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
        vehicle_words = ['jet', 'train']

        #double_word_dict will map double words to the word they should be treated as in our analysis

        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict['baby %s' %animal_word] = animal_word
            self.double_word_dict['adult %s' %animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' %vehicle_word] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'

        self.get_annotations()

    def _load_generated_captions_into_evaluator(self, cap_file, image_id_key, caption_key):

        '''
        Meant to save time so imid_to_objects does not always need to be recomputed.
        '''
        #Read in captions
        self.caps, self.eval_imids = load_generated_captions(cap_file, image_id_key, caption_key)
        assert len(self.caps) == len(self.eval_imids)

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def caption_to_words(self, caption):

        '''
        Input: caption
        Output: MSCOCO words in the caption
        '''

        #standard preprocessing
        words = nltk.word_tokenize(caption.lower())
        tagged_sent = nltk.pos_tag(words)
        lemmas_sent = []
        wnl = WordNetLemmatizer()
        for tag in tagged_sent:
            wordnet_pos = self.get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        # words = [singularize(w) for w in words]
        words = lemmas_sent

        #replace double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
           idxs.append(i)
           double_word = ' '.join(words[i:i+2])
           if double_word in self.double_word_dict:
               double_words.append(self.double_word_dict[double_word])
               i += 2
           else:
               double_words.append(words[i])
               i += 1
        words = double_words

        #toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
        if ('toilet' in words) & ('seat' in words): words = [word for word in words if word != 'seat']

        #get synonyms for all words in the caption
        idxs = [idxs[idx] for idx, word in enumerate(words) \
                if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append(self.inverse_synonym_dict[word])
        #return all the MSCOCO objects in the caption
        return words, node_words, idxs, double_words

    def get_annotations_from_segments(self):
        '''
        Add objects taken from MSCOCO segmentation masks
        '''

        coco_segments = combine_coco_instances(self.coco_path )
        segment_annotations = coco_segments['annotations']

        #make dict linking object name to ids
        id_to_name = {} #dict with id to synsets
        for cat in coco_segments['categories']:
            id_to_name[cat['id']] = cat['name']

        for i, annotation in enumerate(segment_annotations):
            sys.stdout.write("\rGetting annotations for %d/%d segmentation masks"
                              %(i, len(segment_annotations)))
            imid = annotation['image_id']

            node_word = self.inverse_synonym_dict[id_to_name[annotation['category_id']]]
            self.imid_to_objects[imid].append(node_word)
        print("\n")

    def get_annotations_from_captions(self):
        '''
        Add objects taken from MSCOCO ground truth captions
        '''

        coco_caps = combine_coco_captions(self.coco_path)
        caption_annotations = coco_caps['annotations']

        for i, annotation in enumerate(caption_annotations):
            sys.stdout.write('\rGetting annotations for %d/%d ground truth captions'
                              %(i, len(coco_caps['annotations'])))
            imid = annotation['image_id']

            _, node_words, _, _ = self.caption_to_words(annotation['caption'])
            # note here is update, so call get_annotations_from_segments first
            self.imid_to_objects[imid].extend(node_words)
        print("\n")


    def get_annotations(self):

        '''
        Get annotations from both segmentation and captions.  Need both annotation types for CHAIR metric.
        '''

        self.get_annotations_from_segments()
        self.get_annotations_from_captions()
        # deduplicate
        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])

    def compute_chair(self, cap_file, image_id_key, caption_key, debug=False, debug_indices=None):
        '''
        Given ground truth objects and generated captions, determine which sentences have hallucinated words.

        Args:
            cap_file: 描述文件路径
            image_id_key: 图像ID键名
            caption_key: 描述键名
            debug: 是否启用debug模式，输出详细信息
            debug_indices: 需要输出详细信息的样本索引集合（如果为None且debug=True，则输出所有样本）
        '''
        self._load_generated_captions_into_evaluator(cap_file, image_id_key, caption_key)

        imid_to_objects = self.imid_to_objects
        caps = self.caps
        eval_imids = self.eval_imids

        num_caps = 0.
        num_hallucinated_caps = 0.
        hallucinated_word_count = 0.
        coco_word_count = 0.
        len_caps = 0.

        # :add:
        num_recall_gt_objects = 0.
        num_gt_objects = 0.

        output = {'sentences': []}

        # 确定需要输出详细信息的样本
        if debug and debug_indices is None:
            debug_indices = set(range(len(caps)))
        elif not debug:
            debug_indices = set()

        for i in tqdm.trange(len(caps)):
            cap :str = caps[i]
            imid :int = eval_imids[i]

            is_debug = i in debug_indices

            #get all words in the caption, as well as corresponding node word
            # pos = cap.rfind('.')
            # cap = cap[:pos+1]
            words, node_words, idxs, raw_words = self.caption_to_words(cap)

            gt_objects = imid_to_objects[imid]

            # Debug输出：处理过程
            if is_debug:
                print("\n" + "=" * 80)
                print(f"[样本 {i+1}/{len(caps)}] Image ID: {imid}")
                print("=" * 80)
                print(f"原始描述: {cap}")
                print(f"\n[1] 分词和预处理:")
                print(f"  - 原始分词: {raw_words[:20]}..." if len(raw_words) > 20 else f"  - 原始分词: {raw_words}")
                print(f"  - 词形还原后: {words[:20]}..." if len(words) > 20 else f"  - 词形还原后: {words}")
                print(f"  - 识别到的MSCOCO对象: {node_words}")
                print(f"  - 对象数量: {len(node_words)}")

            cap_dict = {'image_id': imid,
                        'caption': cap,
                        'mscoco_hallucinated_words': [],
                        'mscoco_gt_words': list(gt_objects),
                        'mscoco_generated_words': list(node_words),
                        'hallucination_idxs': [],
                        'words': raw_words,
                        'processed_words': words,  # 添加处理后的词
                        'node_words': node_words,  # 添加标准化后的对象名
                        'word_indices': idxs  # 添加词的位置索引
                        }

            # :add:
            cap_dict['metrics'] = {'CHAIRs': 0,
                                   'CHAIRi': 0,
                                   'Recall': 0,
                                   'Len': 0,
                                   }

            #count hallucinated words
            coco_word_count += len(node_words)
            hallucinated = False

            # add
            recall_gt_objects = set()
            hallucinated_details = []  # 详细幻觉信息

            if is_debug:
                print(f"\n[2] Ground Truth 对象:")
                print(f"  - GT对象集合: {sorted(gt_objects)}")
                print(f"  - GT对象数量: {len(gt_objects)}")
                print(f"\n[3] 幻觉检测:")

            for word, node_word, idx in zip(words, node_words, idxs):
                if node_word not in gt_objects:
                    hallucinated_word_count += 1
                    cap_dict['mscoco_hallucinated_words'].append((word, node_word))
                    cap_dict['hallucination_idxs'].append(idx)
                    hallucinated = True
                    hallucinated_details.append({
                        'word': word,
                        'node_word': node_word,
                        'position': idx,
                        'reason': f"'{node_word}' 不在GT对象集合中"
                    })
                    if is_debug:
                        print(f"  ✗ 幻觉: '{word}' -> '{node_word}' (位置: {idx})")
                else:
                    recall_gt_objects.add(node_word)
                    if is_debug:
                        print(f"  ✓ 正确: '{word}' -> '{node_word}' (位置: {idx})")

            #count hallucinated caps
            num_caps += 1
            len_caps += len(raw_words)
            if hallucinated:
               num_hallucinated_caps += 1

            # add
            num_gt_objects += len(gt_objects)
            num_recall_gt_objects += len(recall_gt_objects)

            cap_dict['metrics']['CHAIRs'] = int(hallucinated)
            cap_dict['metrics']['CHAIRi'] = 0.
            cap_dict['metrics']['Recall'] = 0.
            cap_dict['metrics']['Len'] = 0.

            # 添加详细的幻觉信息
            cap_dict['hallucination_details'] = hallucinated_details
            cap_dict['recall_gt_objects'] = list(recall_gt_objects)
            cap_dict['recall_count'] = len(recall_gt_objects)


            if len(words) > 0:
                cap_dict['metrics']['CHAIRi'] = len(cap_dict['mscoco_hallucinated_words'])/float(len(words))

            # add
            if len(gt_objects) > 0:
                cap_dict['metrics']['Recall'] = len(recall_gt_objects) / len(gt_objects)

            # 计算平均长度（以0.01为单位）
            cap_dict['metrics']['Len'] = len(raw_words) * 0.01

            # Debug输出：结果摘要
            if is_debug:
                print(f"\n[4] 结果摘要:")
                print(f"  - 是否包含幻觉: {'是' if hallucinated else '否'}")
                print(f"  - 幻觉对象数量: {len(cap_dict['mscoco_hallucinated_words'])}")
                print(f"  - 正确对象数量: {len(recall_gt_objects)}")
                print(f"  - 总词数: {len(raw_words)}")
                print(f"  - CHAIRs (句子级别): {cap_dict['metrics']['CHAIRs']}")
                print(f"  - CHAIRi (实例级别): {cap_dict['metrics']['CHAIRi']:.4f}")
                print(f"  - Recall (召回率): {cap_dict['metrics']['Recall']:.4f}")
                print(f"  - Len (平均长度): {cap_dict['metrics']['Len']:.4f}")
                print("=" * 80)

            output['sentences'].append(cap_dict)

        chair_s = (num_hallucinated_caps/num_caps)
        chair_i = (hallucinated_word_count/coco_word_count)
        # add
        recall = num_recall_gt_objects / num_gt_objects
        avg_len = (0.01*len_caps/num_caps)

        output['overall_metrics'] = {'CHAIRs': chair_s,
                                     'CHAIRi': chair_i,
                                     'Recall': recall,
                                     'Len': avg_len,}

        return output

def load_generated_captions(cap_file, image_id_key:str, caption_key:str):
    #Read in captions
    # it should be list of dict
    ext = os.path.splitext(cap_file)[-1]
    if ext == '.json':
        caps = json.load(open(cap_file))
    elif ext == '.jsonl':
        caps = [json.loads(s) for s in open(cap_file)]
    else:
        raise ValueError(f'Unspported extension {ext} for cap_file: {cap_file}')

    # list of int
    imids = [obj[image_id_key] for obj in caps]

    # list of str
    caps = [obj[caption_key] for obj in caps]

    return caps, imids

def save_hallucinated_words(cap_file, cap_dict):
    """
    保存详细的CHAIR评估结果到JSON文件

    保存的内容包括：
    - overall_metrics: 总体指标（CHAIRs, CHAIRi, Recall, Len）
    - sentences: 每个样本的详细信息，包括：
      - image_id: 图像ID
      - caption: 原始描述
      - mscoco_gt_words: Ground Truth对象列表
      - mscoco_generated_words: 生成描述中的对象列表
      - mscoco_hallucinated_words: 幻觉对象列表（(word, node_word)元组）
      - hallucination_details: 详细的幻觉信息（包含原因）
      - recall_gt_objects: 正确识别的GT对象
      - recall_count: 正确识别的对象数量
      - processed_words: 处理后的词列表
      - node_words: 标准化后的对象名
      - word_indices: 词的位置索引
      - words: 原始分词结果
      - metrics: CHAIRs, CHAIRi, Recall, Len等指标

    注意：列表字段（如 mscoco_gt_words, mscoco_generated_words, words 等）会以紧凑格式（单行）保存
    """
    # 需要紧凑格式的列表字段（单行显示）
    compact_list_fields = {
        'mscoco_gt_words', 'mscoco_generated_words', 'mscoco_hallucinated_words',
        'words', 'processed_words', 'node_words', 'word_indices',
        'recall_gt_objects', 'hallucination_idxs'
    }

    def format_json_compact(obj, indent_level=0, compact_fields=None):
        """
        自定义JSON格式化函数，对指定的列表字段使用紧凑格式（单行）
        """
        if compact_fields is None:
            compact_fields = set()

        indent = '  ' * indent_level
        next_indent = '  ' * (indent_level + 1)

        if isinstance(obj, dict):
            if not obj:
                return '{}'

            lines = []
            items = list(obj.items())
            for i, (key, value) in enumerate(items):
                # 检查是否是需要紧凑格式的字段
                is_compact = key in compact_fields

                if isinstance(value, (list, tuple)) and is_compact:
                    # 列表字段：紧凑格式（单行）
                    if isinstance(value, tuple):
                        value = list(value)
                    json_value = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                    lines.append(f'{next_indent}"{key}": {json_value}')
                elif isinstance(value, dict):
                    # 嵌套字典：递归处理
                    formatted_value = format_json_compact(value, indent_level + 1, compact_fields)
                    lines.append(f'{next_indent}"{key}": {formatted_value}')
                elif isinstance(value, (list, tuple)):
                    # 其他列表：正常格式（多行）
                    if isinstance(value, tuple):
                        value = list(value)
                    if not value:
                        lines.append(f'{next_indent}"{key}": []')
                    else:
                        list_lines = [f'{next_indent}"{key}": [']
                        for item in value:
                            if isinstance(item, (dict, list)):
                                formatted_item = format_json_compact(item, indent_level + 2, compact_fields)
                                list_lines.append(f'{next_indent}  {formatted_item},')
                            else:
                                json_item = json.dumps(item, ensure_ascii=False)
                                list_lines.append(f'{next_indent}  {json_item},')
                        # 移除最后一个逗号
                        if list_lines[-1].endswith(','):
                            list_lines[-1] = list_lines[-1][:-1]
                        list_lines.append(f'{next_indent}]')
                        lines.append('\n'.join(list_lines))
                else:
                    # 其他类型：正常格式
                    json_value = json.dumps(value, ensure_ascii=False)
                    lines.append(f'{next_indent}"{key}": {json_value}')

            return '{\n' + ',\n'.join(lines) + '\n' + indent + '}'

        elif isinstance(obj, (list, tuple)):
            if isinstance(obj, tuple):
                obj = list(obj)
            if not obj:
                return '[]'
            # 列表：正常格式（多行）
            lines = []
            for item in obj:
                if isinstance(item, (dict, list)):
                    formatted_item = format_json_compact(item, indent_level + 1, compact_fields)
                    lines.append(f'{next_indent}{formatted_item},')
                else:
                    json_item = json.dumps(item, ensure_ascii=False)
                    lines.append(f'{next_indent}{json_item},')
            # 移除最后一个逗号
            if lines and lines[-1].endswith(','):
                lines[-1] = lines[-1][:-1]
            return '[\n' + '\n'.join(lines) + '\n' + indent + ']'

        else:
            # 基本类型：直接JSON编码
            return json.dumps(obj, ensure_ascii=False)

    with open(cap_file, 'w', encoding='utf-8') as f:
        formatted_json = format_json_compact(cap_dict, indent_level=0, compact_fields=compact_list_fields)
        f.write(formatted_json)

def print_metrics(hallucination_cap_dict, quiet=False):
    sentence_metrics = hallucination_cap_dict['overall_metrics']

    for k, v in sentence_metrics.items():
        k_str = str(k).ljust(10)
        v_str = f'{v * 100:.01f}'
        print(k_str, v_str, sep=': ')


def get_chair_evaluator(coco_path, cache_file=None, use_cache=True):
    """
    获取或创建 CHAIR 评估器对象，支持缓存机制

    Args:
        coco_path: COCO annotations 目录路径
        cache_file: 缓存文件路径（如果为 None，则不使用缓存）
        use_cache: 是否使用缓存（如果为 False，即使缓存存在也会重新创建）

    Returns:
        CHAIR: CHAIR 评估器对象
    """
    if use_cache and cache_file and os.path.exists(cache_file):
        try:
            evaluator = pickle.load(open(cache_file, 'rb'))
            print(f"✓ 从缓存加载 CHAIR 评估器: {cache_file}")
            return evaluator
        except Exception as e:
            print(f"⚠️  警告: 加载缓存失败: {e}，将重新创建评估器")

    print(f"正在创建 CHAIR 评估器（这可能需要一些时间）...")
    evaluator = CHAIR(coco_path)

    if cache_file:
        try:
            pickle.dump(evaluator, open(cache_file, 'wb'))
            print(f"✓ CHAIR 评估器已缓存到: {cache_file}")
        except Exception as e:
            print(f"⚠️  警告: 保存缓存失败: {e}")

    return evaluator


def evaluate_chair(cap_file, coco_path, image_id_key="image_id", caption_key="caption",
                   cache_file=None, use_cache=True, save_path=None, verbose=True, debug=False, debug_indices=None):
    """
    计算 CHAIR 指标的高级接口函数

    Args:
        cap_file: 描述文件路径（JSON 或 JSONL 格式）
        coco_path: COCO annotations 目录路径
        image_id_key: 描述文件中图像 ID 的键名（默认："image_id"）
        caption_key: 描述文件中描述的键名（默认："caption"）
        cache_file: 缓存文件路径（可选，用于加速重复评估）
        use_cache: 是否使用缓存（默认：True）
        save_path: 保存详细结果的路径（可选，JSON 格式）
        verbose: 是否输出详细信息（默认：True）
        debug: 是否启用debug模式，输出每个样本的详细处理过程（默认：False）
        debug_indices: 需要输出详细信息的样本索引集合（如果为None且debug=True，则输出所有样本）

    Returns:
        dict: 包含以下键的字典：
            - 'overall_metrics': 总体指标（CHAIRs, CHAIRi, Recall, Len）
            - 'sentences': 每个句子的详细结果
            - 'evaluator': CHAIR 评估器对象（可用于后续评估）
    """
    # 获取或创建评估器
    evaluator = get_chair_evaluator(coco_path, cache_file, use_cache)

    # 计算 CHAIR 指标
    if verbose:
        print(f"\n正在计算 CHAIR 指标...")
        print(f"  描述文件: {cap_file}")
        print(f"  图像 ID 键: {image_id_key}")
        print(f"  描述键: {caption_key}")
        if debug:
            if debug_indices is None:
                print(f"  Debug模式: 启用（输出所有样本的详细信息）")
            else:
                print(f"  Debug模式: 启用（输出 {len(debug_indices)} 个样本的详细信息）")

    results = evaluator.compute_chair(cap_file, image_id_key, caption_key, debug=debug, debug_indices=debug_indices)

    # 打印指标
    if verbose:
        print("\n" + "=" * 80)
        print("CHAIR 评估结果")
        print("=" * 80)
        print_metrics(results)
        print("=" * 80)

    # 保存详细结果
    if save_path:
        save_hallucinated_words(save_path, results)
        if verbose:
            print(f"\n✓ 详细结果已保存到: {save_path}")

    # 返回结果和评估器（评估器可以重复使用）
    results['evaluator'] = evaluator
    return results


def evaluate_chair_from_dict(captions_dict, coco_path, image_id_key="image_id", caption_key="caption",
                             cache_file=None, use_cache=True, save_path=None, verbose=True):
    """
    从字典列表计算 CHAIR 指标（不需要先保存到文件）

    Args:
        captions_dict: 描述字典列表，每个字典包含 image_id 和 caption
        coco_path: COCO annotations 目录路径
        image_id_key: 字典中图像 ID 的键名（默认："image_id"）
        caption_key: 字典中描述的键名（默认："caption"）
        cache_file: 缓存文件路径（可选）
        use_cache: 是否使用缓存（默认：True）
        save_path: 保存详细结果的路径（可选）
        verbose: 是否输出详细信息（默认：True）

    Returns:
        dict: 包含 overall_metrics 和 sentences 的字典
    """
    import tempfile

    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        for item in captions_dict:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        temp_file = f.name

    try:
        # 使用临时文件调用 evaluate_chair
        results = evaluate_chair(
            cap_file=temp_file,
            coco_path=coco_path,
            image_id_key=image_id_key,
            caption_key=caption_key,
            cache_file=cache_file,
            use_cache=use_cache,
            save_path=save_path,
            verbose=verbose
        )
        return results
    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--cap_file", type=str, default='',
                        help="path towards json or jsonl saving image ids and their captions in list of dict.")
    parser.add_argument("--image_id_key", type=str, default="image_id",
                        help="in each dict of cap_file, which key stores image id of coco.")
    parser.add_argument("--caption_key", type=str, default="caption",
                        help="in each dict of cap_file, which key stores caption of the image.")

    parser.add_argument("--cache", type=str, default="chair.pkl",
                        help="pre inited CHAIR evaluator object, for fast loading.")
    parser.add_argument("--coco_path", type=str, default='.../val2014/annotations',
                        help="only use for regenerating CHAIR evaluator object, will be ignored if uses cached evaluator.")

    parser.add_argument("--save_path", type=str, default="...",
                        help="saving CHAIR evaluate and results to json, useful for debugging the caption model.")

    args = parser.parse_args()

    if args.cache and os.path.exists(args.cache):
        evaluator = pickle.load(open(args.cache, 'rb'))
        print(f"loaded evaluator from cache: {args.cache}")
    else:
        print(f"cache not setted or not exist yet, building from scratch...")
        evaluator = CHAIR(args.coco_path)
        pickle.dump(evaluator, open(args.cache, 'wb'))
        print(f"cached evaluator to: {args.cache}")

    cap_dict = evaluator.compute_chair(args.cap_file, args.image_id_key, args.caption_key)

    print_metrics(cap_dict)

    if args.save_path:
        save_hallucinated_words(args.save_path, cap_dict)


# CUDA_VISIBLE_DEVICES=5 python chair.py \
# --cap_file ../POPE-Adv/text_feat/chair-eval/instructblip/ours.jsonl \
# --image_id_key image_id --caption_key caption \
# --coco_path /mnt/petrelfs/share_data/wangjiaqi/mllm-data-alg/COCO_2014/ori/annotations_trainval2014/annotations/ \
# --save_path ../POPE-Adv/text_feat/chair-eval/instructblip/ours_outputs.json