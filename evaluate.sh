# JSON_ONLY
python graph_evaluation_fn.py --lg_folder data/Crohme_all_LGs --json_folder data/Test2019_primitive_json --inkml_folder data/Test2019 --mode JSON_ONLY --output input_graph_test2019_json_only.csv
python graph_evaluation_fn.py --lg_folder data/Crohme_all_LGs --json_folder data/Test2016_primitive_json --inkml_folder data/Test2016 --mode JSON_ONLY --output input_graph_test2016_json_only.csv
python graph_evaluation_fn.py --lg_folder data/Crohme_all_LGs --json_folder data/Test2014_primitive_json --inkml_folder data/Test2014 --mode JSON_ONLY --output input_graph_test2014_json_only.csv
python graph_evaluation_fn.py --lg_folder data/Crohme_all_LGs --json_folder data/Train_primitive_json --inkml_folder data/Train --mode JSON_ONLY --output input_graph_train_json_only.csv

# JSON_LOS
python graph_evaluation_fn.py --lg_folder data/Crohme_all_LGs --json_folder data/Test2019_primitive_json --inkml_folder data/Test2019 --mode JSON_LOS --output input_graph_test2019_json_los.csv
python graph_evaluation_fn.py --lg_folder data/Crohme_all_LGs --json_folder data/Test2016_primitive_json --inkml_folder data/Test2016 --mode JSON_LOS --output input_graph_test2016_json_los.csv
python graph_evaluation_fn.py --lg_folder data/Crohme_all_LGs --json_folder data/Test2014_primitive_json --inkml_folder data/Test2014 --mode JSON_LOS --output input_graph_test2014_json_los.csv
python graph_evaluation_fn.py --lg_folder data/Crohme_all_LGs --json_folder data/Train_primitive_json --inkml_folder data/Train --mode JSON_LOS --output input_graph_train_json_los.csv

# LOS only
python graph_evaluation_fn.py --lg_folder data/Crohme_all_LGs --json_folder data/Test2019_primitive_json --inkml_folder data/Test2019 --mode LOS_ONLY --output input_graph_test2019_los_only.csv
python graph_evaluation_fn.py --lg_folder data/Crohme_all_LGs --json_folder data/Test2016_primitive_json --inkml_folder data/Test2016 --mode LOS_ONLY --output input_graph_test2016_los_only.csv
python graph_evaluation_fn.py --lg_folder data/Crohme_all_LGs --json_folder data/Test2014_primitive_json --inkml_folder data/Test2014 --mode LOS_ONLY --output input_graph_test2014_los_only.csv
python graph_evaluation_fn.py --lg_folder data/Crohme_all_LGs --json_folder data/Train_primitive_json --inkml_folder data/Train --mode LOS_ONLY --output input_graph_train_los_only.csv