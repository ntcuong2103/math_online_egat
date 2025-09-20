from data import GraphData, map_lg_to_json, map_lg_to_json_sym
import os
from tqdm import tqdm

def build_file_dict(folder, extension):
    """
    Walk through the folder (including subfolders) and build a dictionary mapping 
    the lowercased base name (without extension) to the full file path.
    For JSON files, only include files if their path contains one of the allowed subfolders.
    """
    file_dict = {}
    allowed_json_folders = {"test2019_primitive_json", "test2016_primitive_json", 
                              "test2014_primitive_json", "train_primitive_json"}
    
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(extension):
                if extension.lower() == '.json':
                    path_parts = [part.lower() for part in root.split(os.sep)]
                    if not any(allowed in path_parts for allowed in allowed_json_folders):
                        continue
                base_name = os.path.splitext(file)[0].lower()
                file_dict[base_name] = os.path.join(root, file)
    return file_dict


def process_dataset(lg_folder, json_folder, inkml_folder, mode='JSON_LOS'):
    """
    Process the dataset and evaluate the files in the given folders.
    """
    # Build dictionaries for LG and JSON files.
    lg_files = build_file_dict(lg_folder, '.lg')
    json_files = build_file_dict(json_folder, '.json')
    inkml_files = build_file_dict(inkml_folder, '.inkml')
    
    # Get the sorted list of base names from the LG dictionary.
    lg_keys = sorted(lg_files.keys())
    
    # Prepare list of matching file pairs.
    matched_pairs = []
    for base_name in lg_keys:
        if (base_name in json_files) and (base_name in inkml_files):
            matched_pairs.append((lg_files[base_name], json_files[base_name], inkml_files[base_name], base_name))
    
    total_pairs = len(matched_pairs) 
    print(f"Total file pairs to process: {total_pairs}")
    
    processed = 0
    results = {}
    for lg_file_path, json_file_path, inkml_file_path, base_name in tqdm(matched_pairs, desc="Processing file pairs"):
        try:
            results[base_name] = evaluate_file_pair(lg_file_path, json_file_path, inkml_file_path, mode)

        except Exception as e:
            print(f"Skipping file '{base_name}' due to error: {e}")

    return results

def extract_edges(relations, label=False):
    edges = set()
    for key, value in relations.items():
        #if value['relation']!='No': # remove the 'No' relation in JSON_LOS_edges 
        if label:
            edges.add((key, value['relation']))
        else:
            edges.add(key)
    return edges

# [ Seg ], [ Seg + Class ], [ Edge ], [ Edge missing ], [ Edge redundant ],

def evaluate_file_pair(lg_file_path, json_file_path, inkml_file_path, mode='JSON_LOS'):    
    # mode LOS & JSON
    graph_json = GraphData()
    graph_json.readJson(json_file_path)
    if mode == 'JSON_ONLY':
        pass    
    elif mode == 'JSON_LOS':
        graph_json.filterEdges(graph_json.getEdgesByLOS(inkml_file_path))
    elif mode == 'LOS_ONLY':
        graph_json.Relations = graph_json.getEdgesByLOS(inkml_file_path)

    json_edges = extract_edges(graph_json.Relations)

    # LG
    graph_lg = GraphData()
    graph_lg.readLGs(lg_file_path)

    lg_edges = extract_edges(graph_lg.Relations)

    # Compare edges 
    # between JSON_LOS and LG
    common_edges = json_edges.intersection(lg_edges)
    missing_edges = len(lg_edges) - len(common_edges)
    redundant_edges = len(json_edges) - len(common_edges)
    # no missing edges
    coverage = missing_edges == 0

    # evaluate labels
    json_edges_label = extract_edges(graph_json.Relations, label=True)
    lg_edges_label = extract_edges(graph_lg.Relations, label=True)
    common_edges_label = json_edges_label.intersection(lg_edges_label)


    # evaluate symbols
    json_symbols = set(graph_json.Symbols.keys())
    lg_symbols = set(graph_lg.Symbols.keys())
    common_symbols = json_symbols.intersection(lg_symbols)

    seg_correct = len(common_symbols)

    # symbols with labels
    json_symbols_label = set([(key, value['label']) for key, value in graph_json.Symbols.items()])
    lg_symbols_label = set([(key, map_lg_to_json_sym.get(value['label'], value['label'])) for key, value in graph_lg.Symbols.items()])
    common_symbols_label = json_symbols_label.intersection(lg_symbols_label)

    seg_label_correct = len(common_symbols_label)
    
    return {
        'seg_correct': seg_correct,
        'seg_label_correct': seg_label_correct,
        'lg_symbols': len(lg_symbols),
        'missing_edges': missing_edges,
        'redundant_edges': redundant_edges,
        'common_edges_label': len(common_edges_label),
        'lg_edges': len(lg_edges),
        'coverage': coverage
    }



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lg_folder', type=str, default='data/Crohme_all_LGs')
    parser.add_argument('--json_folder', type=str, default='data/Test2019_primitive_json')
    parser.add_argument('--inkml_folder', type=str, default='data/Test2019')
    parser.add_argument('--mode', type=str, default='LOS_JSON')
    parser.add_argument('--output', type=str, default='results.json')
    args = parser.parse_args()


    results = process_dataset(args.lg_folder, args.json_folder, args.inkml_folder, args.mode)
    # write to csv with key
    import csv
    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = ['file_id', 'seg_correct', 'seg_label_correct', 'lg_symbols', 'missing_edges', 'redundant_edges', 'common_edges_label', 'lg_edges', 'coverage']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key, result in results.items():
            writer.writerow({'file_id': key, **result})