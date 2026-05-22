import click
import time
import os
import shutil
import hashlib
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from job_ini import JobIni
from data import MVTecDataset, mvtec_classes
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

ALL_CLASSES = mvtec_classes()
ALLOWED_METHODS = ["spade", "padim", "patchcore"]
DEFAULT_FEATURE_INDICES = {
	"spade" : "1,2,3,-1",
	"padim" : "1,2,3",
    "patchcore":"2,3",
}

def run_model(jobini, method: str, backbone: str,resize_method: str, out_indices: tuple, cls: str, dataset_dir: str,results_dir: str,image_size: int, f_coreset: float,coreset_eps: float):
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
        model = PatchCore(
            f_coreset=f_coreset, 
            coreset_eps=coreset_eps, 
            backbone_name=backbone,
            out_indices=out_indices,
            results_dir=results_dir,
            image_size=image_size,
        )
    # model = model.to(device)
    print(f"\n█│ Running {method} on {cls} dataset.")
    print(  f" ╰{'─'*(len(method)+len(cls)+23)}\n")
    train_ds, test_ds = MVTecDataset(cls,size=image_size,dataset_dir=dataset_dir,resize_method=resize_method).get_dataloaders()
    jobini.set_exec_progress(10)

    print("   Training ...")
    model.fit(train_ds)
    jobini.set_exec_progress(50)
    
    print("   Testing ...")
    image_rocauc, pixel_rocauc = model.evaluate(test_ds)
    jobini.set_exec_progress(90)

    print(f"\n   ╭{'─'*(len(cls)+15)}┬{'─'*20}┬{'─'*20}╮")
    print(  f"   │ Test results {cls} │ image_rocauc: {image_rocauc:.2f} │ pixel_rocauc: {pixel_rocauc:.2f} │")
    print(  f"   ╰{'─'*(len(cls)+15)}┴{'─'*20}┴{'─'*20}╯")
    results[cls] = [float(image_rocauc), float(pixel_rocauc)]
        
    image_results = [v[0] for _, v in results.items()]
    average_image_roc_auc = sum(image_results)/len(image_results)
    image_results = [v[1] for _, v in results.items()]
    average_pixel_roc_auc = sum(image_results)/len(image_results)

    total_results = {
        "per_class_results": results,
        "average image rocauc": average_image_roc_auc,
        "average pixel rocauc": average_pixel_roc_auc,
        "model parameters": model.get_parameters(),
    }
    run_info = {
        "fmap_size": model.get_fmap_size(),
        "image_shape": model.get_image_shape(),
    }
    return total_results,run_info

def md5_file(path):
    with open(path, 'rb') as f:
        hashval = hashlib.new('md5', f.read()).hexdigest()
        return hashval[0:8]
        
def output_clear(dataset_dir):
    model_txt = os.path.join(dataset_dir, "model.txt")    
    if os.path.exists(model_txt):
        os.remove(model_txt)
     
def output_result(dataset_dir,result_dir,output_name):
    #copy patch_lib.ts
    src_path = os.path.join(result_dir, "patch_lib.ts")
    dst_name = output_name.replace('[md5]',md5_file(src_path))
    dst_path = os.path.join(result_dir, dst_name)
    shutil.move(src_path, dst_path)
    #写入 model.txt 文件
    model_txt = os.path.join(dataset_dir, "model.txt")
    f = open(model_txt, "w")
    f.write(dst_name)
    f.close()
    
@click.command()
@click.argument("method")
@click.option("--backbone", default="resnet18", required=False,help="backbone name")
@click.option("--resize_method", default="cv2", required=False,help="resize method. trasform,cv2")
@click.option("--feature_indices", default="", required=False,type=str,help="feature indices")
@click.option("--dataset", default="not all", required=False,help="dataset, defaults to all datasets.")
@click.option("--dataset_dir", default="./datasets", required=False,type=str,help="dataset dir")
@click.option("--result_dir", default="./results", required=False,type=str,help="result dir")
@click.option("--image_size", default=224, required=False,type=int,help="training image size")
@click.option("--f_coreset", default=0.10, required=False,type=float,help="fraction the number of training samples")
@click.option("--coreset_eps", default=0.90, required=False,type=float,help="sparse projection parameter")
def cli_interface(method:str, backbone: str,resize_method: str,feature_indices: tuple, dataset: str, dataset_dir: str, result_dir: str, image_size: int, f_coreset: float,coreset_eps: float):
    #尺寸确保4的倍数
    image_size = int(image_size/4)*4
    
    #清空输出文件夹内容
    output_clear(dataset_dir)
    #进度设为0
    jobini = JobIni(dataset_dir)
    jobini.set_exec_progress(0)
    
    #设置torch.hub路径
    curdir = os.path.dirname(os.path.abspath(__file__))
    hub_dir = os.path.join(os.path.dirname(curdir),'hub')
    torch.hub.set_dir(hub_dir)
    
    #检查方法
    method = method.lower()
    assert method in ALLOWED_METHODS, f"Select from {ALLOWED_METHODS}."
    
    #特征索引
    if feature_indices == "":
        feature_indices = DEFAULT_FEATURE_INDICES[method]
    out_indices=tuple([int(i) for i in feature_indices.split(',')])

    #输出目录
    curr_time = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
    result_name = '{0}_{1}_{2}_{3}_{4}'.format(curr_time,method,backbone,feature_indices,image_size)
    result_dir = os.path.join(result_dir, dataset )
    result_dir = os.path.join(result_dir, result_name )
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    #运行模型
    total_results,run_info = run_model(jobini,method, backbone, resize_method ,out_indices, dataset,dataset_dir,result_dir,image_size, f_coreset,coreset_eps)
    
    #输出结果
    print_and_export_results(total_results, method,result_dir)
    
    #保存模型
    backbone_name = backbone.replace('_','-')
    fmap_size = '%dx%d' % (run_info['fmap_size'][0], run_info['fmap_size'][1])
    image_info = "%dx%dx%d" %(run_info['image_shape'][0],run_info['image_shape'][1],run_info['image_shape'][2])
    output_name = f'{method}_jobno_{resize_method}_{backbone_name}_23_{fmap_size}_{image_info}_fp32_[md5].ts'
    output_result(dataset_dir,result_dir,output_name)
    
    jobini.set_exec_progress(100)
        
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cli_interface()
