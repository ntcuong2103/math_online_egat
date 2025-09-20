# Description: Create pickle data for training and testing
python data.py --input_jsons=data/Test2014_primitive_json --input_lgs=data/Crohme_all_LGs --output=data_pkls/base/test2014.pkl
python data.py --input_jsons=data/Test2016_primitive_json --input_lgs=data/Crohme_all_LGs --output=data_pkls/base/test2016.pkl
python data.py --input_jsons=data/Test2019_primitive_json --input_lgs=data/Crohme_all_LGs --output=data_pkls/base/test2019.pkl
python data.py --input_jsons=data/Train_primitive_json --input_lgs=data/Crohme_all_LGs --output=data_pkls/base/train.pkl
# With LOS
python data.py --input_jsons=data/Test2014_primitive_json --input_lgs=data/Crohme_all_LGs --input_inkmls=data/Test2014 --use_los --output=data_pkls/los/test2014.pkl
python data.py --input_jsons=data/Test2016_primitive_json --input_lgs=data/Crohme_all_LGs --input_inkmls=data/Test2016 --use_los --output=data_pkls/los/test2016.pkl
python data.py --input_jsons=data/Test2019_primitive_json --input_lgs=data/Crohme_all_LGs --input_inkmls=data/Test2019 --use_los --output=data_pkls/los/test2019.pkl
python data.py --input_jsons=data/Train_primitive_json --input_lgs=data/Crohme_all_LGs --input_inkmls=data/Train --use_los --output=data_pkls/los/train.pkl
