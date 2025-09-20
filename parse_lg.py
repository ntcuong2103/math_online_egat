import os
import random
import numpy as np

class Object:
    def __init__(self, *args,**kwargs):
        if len(args) > 1:
            self.id = args[0]
            self.sym = args[1]
            self.prob = args[2]
            self.strokes = args[3]
        else:
            line = args[0]
            line = line.split(', ')
            self.id = line[1]
            self.sym = line[2]
            self.prob = float(line[3])
            self.strokes = sorted([int(s) for s in line[4:]])
    
    def write_lg(self):
        return ', '.join(['O', str(self.id), self.sym, '1.0'] + [str(s) for s in self.strokes])

class Relation:
    def __init__(self, *args,**kwargs):
        if len(args) > 1:
            self.id1 = args[0]
            self.id2 = args[1]
            self.relation = args[2]
            self.prob = args[3]
        else:
            line = args[0]
            line = line.split(', ')
            self.id1 = line[1]
            self.id2 = line[2]
            self.relation = line[3]
            self.prob = float(line[4])
    def write_lg(self):
        return ', '.join(['EO', str(self.id1), str(self.id2), self.relation, '1.0'])

class Tree:
    def __init__(self, obj):
        self.value = obj
        self.childs = []
        self.relations = []
    def add_child(self, child, relation):
        self.childs.append(child)
        self.relations.append(relation)

def build_tree(objects, relations):
    nodes = [Tree(obj) for obj in objects]
    id2node = {node.value.id:node for node in nodes}

    if len(relations) == 0 or len(nodes) == 0: return None

    for rel in relations:
        # id 1 -> node, id 2 -> childs
        if rel.id1 in id2node.keys() and rel.id2 in id2node.keys():
            id2node[rel.id1].add_child(id2node[rel.id2], rel.relation)
    
    # find root
    id_list = [(rel.id1, rel.id2) for rel in relations]
    id1_list, id2_list = list(zip(*id_list))
    all_roots = [id for id in id1_list if id not in id2_list]
    
    # from collections import Counter
    # # all roots
    # id1_count = Counter(id1_list)
    # all_roots = [id for id in id1_count if id1_count[id] > 1] 
    edged_nodes = list(set(id1_list + id2_list))
    single_nodes = [id for id in id2node.keys() if id not in edged_nodes]

    return [id2node[root_id] for root_id in all_roots], [id2node[s_id] for s_id in single_nodes], id2node

def extract_relations(root_node):
    relations = []

    for child, relation in zip(root_node.childs, root_node.relations):
        relations.append(Relation(root_node.value.id, 
                                            child.value.id,
                                            relation,
                                            1.0))   
        relations.extend(extract_relations(child))
    return relations

def extract_objects_relations(root_node, id2node):
    nodes = list(id2node.values())
    objects = [node.value for node in nodes]
    relations = extract_relations(root_node)
       
    return objects, relations



def print_tree(root_node, level=0):
    print(''.join(['\t']*level) + root_node.value.sym)
    for child in root_node.childs:
        print_tree(child, level+1)

def extract_path(root_node, in_path):
    in_path += [root_node]
    if len(root_node.childs) == 0: return [in_path]
    
    path = []
    for child, relation in zip(root_node.childs, root_node.relations):
        path.extend(extract_path(child, in_path + [relation]))
    return path

def extract_path_with_nrel(root_node, in_path):
    in_path += [root_node]
    if len(root_node.childs) == 0: return [in_path]

    path = []
    # for child, relation in zip(root_node.childs, root_node.relations):
    #     path.extend(extract_path_with_nrel(child, in_path + [relation]))
    for id in range(len(root_node.childs)):
        path.extend(extract_path_with_nrel(root_node.childs[id], in_path + [root_node.relations[id]]))
        # nrels
        for id_c in range(len(root_node.childs)):
            if id_c != id:
                path.extend(extract_path_with_nrel(root_node.childs[id], in_path + [root_node.relations[id_c], root_node.childs[id_c], 'NoRel']))
    return path

def extract_ordered_path(objects, relations):
    sorted_objects = sorted(objects, key=lambda obj: obj.strokes[0])
    if len(relations) == 0 or len(objects) == 0: return []

    id_pairs = {relation.id1 + relation.id2:relation.relation for relation in relations}

    output = []
    for obj1, obj2 in zip(sorted_objects[:-1], sorted_objects[1:]):
        output.append(obj1)
        if (obj1.id + obj2.id) in id_pairs:
            output.append(id_pairs[obj1.id + obj2.id])
        else:
            output.append('NoRel')
    
    output.append(sorted_objects[-1])
    return output

def extract_random_walks(root_node):
    if len(root_node.childs) == 0: return [root_node.value]
    if len(root_node.childs) == 1: return [root_node.value] + extract_random_walks(root_node.childs[0])

    path = []
    # random insert root
    if random.random() < 1.0:
        root_idx = random.randint(0, len(root_node.childs))
    else:
        root_idx = 0
    id = 0
    for child in random.sample(root_node.childs, len(root_node.childs)):
        if id == root_idx:
            path += [root_node.value]
        path += extract_random_walks(child)
        id +=1
    if id == root_idx:
        path += [root_node.value]
    # print (len(path))
    return path


def unique(in_list):
    unique_list = []
    for x in in_list:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def extract_random_path(objects, relations, root_node):
    objects_path = unique(extract_random_walks(root_node))

    # assert (len(objects) == len(objects_path)), "{} # {}".format(len(objects), len(objects_path))
    # if len(objects) != len(objects_path): return []
    if len(relations) == 0 or len(objects) == 0: return []

    id_pairs = {relation.id1 + '_' + relation.id2: relation.relation for relation in relations}

    output = []
    for obj1, obj2 in zip(objects_path[:-1], objects_path[1:]):
        output.append(obj1)
        if (obj1.id + '_' + obj2.id) in id_pairs:
            output.append(id_pairs[obj1.id + '_' + obj2.id])
        else:
            output.append('NoRel')

    output.append(objects_path[-1])
    return output

def print_path(path):
    path_str = []
    for node in path:
        if isinstance(node, Tree):
            path_str += [node.value.sym]
        elif isinstance(node, Object):
            path_str += [node.sym]
        else:
            path_str += [node]
    # print(' '.join(path_str))
    return path_str


def get_strokes(path):
    path_strks = []
    for node in path:
        if isinstance(node, Tree):
            path_strks.extend(node.value.strokes)
        elif isinstance(node, Object):
            path_strks.extend(node.strokes)
    return path_strks

def get_rel_masks(path, strk_lens, str_ids):
    masks = [0] * sum(strk_lens[id] for id in str_ids)
    idx = 0
    for node in path:
        if isinstance(node, Tree):
            idx += sum([strk_lens[stroke] for stroke in node.value.strokes])
            if idx < len(masks):
                masks[idx] = 1
        if isinstance(node, Object):
            idx += sum([strk_lens[stroke] for stroke in node.strokes])
            if idx < len(masks):
                masks[idx] = 1
    return masks

def parse_lg(input_fn):
    lines = [line.strip() for line in open(input_fn).readlines()]
    objects = [Object(line) for line in lines if line.startswith('O')]
    relations = [Relation(line) for line in lines if line.startswith('EO') or line.startswith('R')]

    return objects, relations

def write_lg(output_fn, objects, relations):
    lg_str = []

    for obj in objects:
        lg_str += [obj.write_lg()]
    for rel in relations:
        lg_str += [rel.write_lg()]

    with open(output_fn, 'w') as f:
        f.writelines([line + '\n' for line in lg_str])

def parse_folder(input_folder, output_fn):
    from multiprocessing import Pool
    fns = [f.path for f in os.scandir(input_folder) if f.name.endswith(".lg")][:]

    from tqdm import tqdm
    # with Pool(4) as p:
        # roots = p.map(parse_lg, fns)
    # roots = [parse_lg(fn)[0] for fn in tqdm(fns)]
    # objects = [parse_lg(fn)[1] for fn in tqdm(fns)]
    pass

def lg_to_dot(input_fn, output_fn):
    objects, relations = parse_lg(input_fn)

    obj2id = {obj.id:str(i) for i, obj in enumerate(objects)}

    dot_str = []
    dot_str += ["strict digraph G {", "    rankdir=LR;",  "    node [shape=record, width=.1]"]
    
    for obj in objects:
        dot_str += ["    {} [id = \"{}\", label=\"{}\"];".format(obj2id[obj.id], ", ".join([str(s) for s in obj.strokes]), obj.sym.replace('\\', '\\\\'))]

    for rel in relations:
        if rel.relation in ['Above', 'Sup']:
            dot_str += ["    {}:ne -> {} [label=\"{}\"];".format(obj2id[rel.id1], obj2id[rel.id2], rel.relation)]
        elif rel.relation in ['Below', 'Inside', 'Sub']:
            dot_str += ["    {}:se -> {} [label=\"{}\"];".format(obj2id[rel.id1], obj2id[rel.id2], rel.relation)]
        elif rel.relation in ['Right', 'NoRel']:
            dot_str += ["    {} -> {} [label=\"{}\", weight=2];".format(obj2id[rel.id1], obj2id[rel.id2], rel.relation)]
    dot_str += ["}"]

    with open(output_fn, 'w') as f:
        f.writelines([line + '\n' for line in dot_str])

def dot_to_lg(input_fn, output_fn):
    import re
    out_ln = []
    for line in open(input_fn).readlines():
        if re.match(r'^\s*[0-9]+\s*\[.*\]', line):
            out_ln += [', '.join(['O', 
            re.search(r'^\s*([0-9]+)', line).group(1), 
            re.search(r'label\s*=\s*\"(.+?)\"', line).group(1).replace('\\\\','\\'),
            str(1.0),
            re.search(r'id\s*=\s*\"(.+?)\"', line).group(1),
            ])]
        
        if re.match(r'^\s*[0-9]+.*\s*->\s*[0-9]+\s*\[.*\]', line):
            search = re.search(r'^\s*([0-9]+).*\s*->\s*([0-9]+)\s*\[.*\]', line)
            out_ln += [', '.join(['EO', 
            search.group(1),
            search.group(2), 
            re.search(r'label\s*=\s*\"(.+?)\"', line).group(1),
            str(1.0),
            ])]
    
    with open(output_fn, 'w') as f:
        f.writelines([line + '\n' for line in out_ln])


def normalize_lg(input_fn, output_fn):
    objects, relations = parse_lg(input_fn)

    # sum, lim: Above, Below
    query_idx = [obj.id for obj in objects if obj.sym in ['\\sum', '\\lim'] ]

    for rel in relations:
        if rel.id1 in query_idx:
            if rel.relation == 'Sup':
                rel.relation = 'Above'
            elif rel.relation == 'Sub':
                rel.relation = 'Below'

    # frac
    query_idx = [obj.id for obj in objects if obj.sym in ['-'] ]
    query_idx_above = []
    query_idx_below = []
    for rel in relations:
        if rel.id1 in query_idx:
            if rel.relation == 'Above':
                query_idx_above.append(rel.id1)
            elif rel.relation == 'Below':
                query_idx_below.append(rel.id1)
    frac_idx = [id for id in query_idx_above if id in query_idx_below]
    for obj in objects:
        if obj.id in frac_idx:
            obj.sym = '\\frac'

    write_lg(output_fn, objects, relations)

if __name__ == "__main__":
    # normalize_lg('/home/tuancuong/data/math/ilabo2021/WithGT/LGs/Crohme_all/23_em_68.lg', 
    # '/home/tuancuong/data/math/ilabo2021/WithGT/norm_frac_LGs/Crohme_all/23_em_68.lg')
    # exit()

    input_folder = r'/home/tuancuong/data/math/ilabo2021/WithGT/LGs'
    input_fns = []
    for dirpath, dirnames, filenames in os.walk(input_folder):
        input_fns.extend([os.path.join(dirpath, filename) for filename in filenames if filename.endswith(".lg")])

    output_folder = r'/home/tuancuong/data/math/ilabo2021/WithGT/norm_frac_LGs'

    for input_fn in input_fns:
        # base_fn = input_fn.replace(input_folder, '').replace('.dot', '.lg')
        base_fn = input_fn.replace(input_folder, '')
        output_fn = output_folder + base_fn
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)
        try:
            normalize_lg(input_fn, output_fn)
        except:
            print('error %s'%(input_fn))
    
    exit()

    # lg_to_dot(r'D:\database\Math\ilabo_LGs\Metamoji\5-004_1-joko-1-id1.lg', 'output.dot')
    lg_to_dot(r'temp.lg', 'output.dot')

    # dot_to_lg('/home/tuancuong/data/CROHME/valid/dot/18_em_3.dot', '18_em_3.lg')

    exit()
    objects, relations = parse_lg('/home/tuancuong/data/CROHME/valid/TestEM2014_LG/18_em_3.lg')
    all_roots = build_tree(objects, relations)
    exit()
    # parse_folder('valid/TestEM2014_LG')
    import pickle
    (fns, roots) = pickle.load(open('val_tree.pkl', 'rb'))
    paths = [extract_path(root, []) for root in roots if root != None]
    path_vocab = []
    for path_fn in paths:
        for path in path_fn:
            path_vocab.extend(print_path(path))
    print(list(sorted(set(path_vocab))))
    path_vocab = list(set(path_vocab))
    vocab = [line.strip() for line in open('vocab/crohme/vocab_syms.txt').readlines()]
    print(sorted([v for v in path_vocab if v not in vocab]))
    pass
