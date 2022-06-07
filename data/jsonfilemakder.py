import json
import os
import pandas as pd
from pandas import json_normalize
import argparse

from tqdm import tqdm



def main(args):
    path = os.path.dirname(os.path.abspath(__file__))
    path = path + args.mode + '/JPEGImages'
    print(path)
    json_list=[_ for _ in os.listdir(path) if _.endswith(r'.json')]
    print("file len: " , len(json_list))
    
    concat_result = []
    
    # JSON 파일 불러와서 변수에 저장하기
    for (i, dt) in enumerate(json_list):
        file = open(path+f"/{dt}","r",encoding='utf-8')
        # print(json.load(file))
        globals()['json_'+ str(i)] = json.load(file)
        
    # 전체 JSON 하나로 합치기
    for (i, dt) in enumerate(json_list):
        concat_result.append(globals()['json_'+ str(i)])   
    
    print("concat len: " , len(concat_result))
    
    if len(concat_result) != len(json_list):
        print("concat error !!")
        exit(1)
    
    file_path = "./"+ args.mode + ".json"
    with open(file_path, 'w') as outfile:
        json.dump(concat_result, outfile, indent=4) 
    
    df = json_normalize(concat_result)
    df = df.sort_values(by=['file_name'], axis=0)
    df = df.reset_index(drop=True)
    df.to_csv('./'+args.mode +'.csv', index=False)

    print("Done Make files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='/naverboostcamp_train')
    parser.add_argument("--measurement", type=str, default='all')
    args = parser.parse_args()
    main(args)