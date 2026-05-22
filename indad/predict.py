import click
import time
import os
import shutil
import hashlib
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from job_ini import JobIni
from data import MVTecDataset
from models import SPADE, PaDiM, PatchCore
from utils import print_and_export_results, write_file

from typing import List

# seeds
import torch
import random
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

import warnings # for some torch warnings regarding depreciation
warnings.filterwarnings("ignore")

ALLOWED_METHODS = ["spade", "padim", "patchcore"]
DEFAULT_FEATURE_INDICES = {
	"spade" : "1,2,3,-1",
	"padim" : "1,2,3",
    "patchcore":"2,3",
}
# patchcore_jobno_resnet18_23_28x50_3x396x224_fp32_ee23f7.ts
def parse_model_info(model_path):
    file_name = os.path.basename(model_path)
    file_name = file_name.split('.')[0]
    file_info = file_name.split('_')
    if len(file_info) != 9:
        raise ValueError
    
    method = file_info[0]
    jobno = file_info[1]
    resize_method = file_info[2]
    backbone = file_info[3]
    out_indices = file_info[4]
    fmap_size = file_info[5]
    image_shape = file_info[6]
    precision = file_info[7]
    md5 = file_info[8]
    if out_indices == '23':
        out_indices = [2,3]
    else:
        raise ValueError
    backbone = backbone.replace('-','_')
    fmap_size = [int(i) for i in fmap_size.split('x')]
    image_shape = [int(i) for i in image_shape.split('x')]
    
    return {
        "method": method,
        "resize_method": resize_method,
        "jobno": jobno,
        "backbone": backbone,
        "out_indices": out_indices,
        "fmap_size": fmap_size,
        "image_size": image_shape[1]
        }
    

def predict_model(model_path,dataset,dataset_dir,results_dir):
    model_info = parse_model_info(model_path)
    method = model_info['method']
    resize_method = model_info['resize_method']
    backbone = model_info['backbone']
    out_indices = model_info['out_indices']
    fmap_size = model_info['fmap_size']
    image_size = model_info['image_size']
    
    results = {}
    if method == "spade":
        model = SPADE(
            k=50,
            backbone_name=backbone,
            out_indices=out_indices,
            results_dir=results_dir,
            image_size=image_size,
        )
    elif method == "padim":
        model = PaDiM(
            d_reduced=350,
            backbone_name=backbone,
            out_indices=out_indices,
            results_dir=results_dir,
            image_size=image_size,
        )
    elif method == "patchcore":
        f_coreset = 0.1
        coreset_eps = 0.9
        model = PatchCore(
            f_coreset=f_coreset, 
            coreset_eps=coreset_eps, 
            backbone_name=backbone,
            out_indices=out_indices,
            results_dir=results_dir,
            image_size=image_size,
        )
    else:
        raise ValueError

    train_ds, test_ds = MVTecDataset(dataset,size=image_size,dataset_dir=dataset_dir,resize_method=resize_method).get_dataloaders()
    print("   Loading ...")
    model.load(model_path,fmap_size)
    
    #print("   Fitting ...")
    #model.fit(train_ds)

    
    print("   Testing ...")
    image_rocauc, pixel_rocauc = model.evaluate(test_ds)

    print(  f"   │ Test results │ image_rocauc: {image_rocauc:.2f} │ pixel_rocauc: {pixel_rocauc:.2f} │")
    results = [float(image_rocauc), float(pixel_rocauc)]
        
    image_results = [results[0]]
    average_image_roc_auc = sum(image_results)/len(image_results)
    image_results = [results[1]]
    average_pixel_roc_auc = sum(image_results)/len(image_results)

    total_results = {
        "per_class_results": results,
        "average image rocauc": average_image_roc_auc,
        "average pixel rocauc": average_pixel_roc_auc,
        "model parameters": model.get_parameters(),
    }
    return total_results
 
@click.command()
@click.argument("model_path")
@click.option("--dataset", default="", required=True,type=str,help="dataset name")
@click.option("--dataset_dir", default="./datasets", required=False,type=str,help="dataset dir")
@click.option("--results_dir", default="./results-predict", required=False,type=str,help="predict results dir")
def cli_interface(model_path:str, dataset: str, dataset_dir: str, results_dir: str):
    #设置torch.hub路径
    curdir = os.path.dirname(os.path.abspath(__file__))
    hub_dir = os.path.join(os.path.dirname(curdir),'hub')
    torch.hub.set_dir(hub_dir)
      
    #输出目录
    curr_time = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
    results_dir = os.path.join(results_dir, curr_time )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    #运行模型
    total_results = predict_model(model_path,dataset,dataset_dir,results_dir)
            
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cli_interface()
