import click
import time
import os
import yaml
import codecs
import shutil
import hashlib
import sys
import json
import zipfile
import cv2

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

def run_model(jobini, method: str, backbone: str, resize_method:str,out_indices: tuple, cls: str, dataset_dir: str,results_dir: str,image_size: int, f_coreset: float,coreset_eps: float, max_feature_count: int=0, start_pos: int=0, end_pos: int=0, score_normalization=None):
    results = {}
    score_normalization = score_normalization or {}
    if method == "spade":
        backbone_name = backbone if len(backbone) > 0 else "wide_resnet50_2" 
        model = SPADE(
            k=50,
            backbone_name=backbone_name,
            out_indices=out_indices,
            results_dir=results_dir,
            image_size=image_size,
            max_feature_count=max_feature_count,
            jobini=jobini
        )
    elif method == "padim":
        backbone_name = backbone if len(backbone) > 0 else "resnet18" 
        model = PaDiM(
            d_reduced=350,
            backbone_name=backbone_name,
            out_indices=out_indices,
            results_dir=results_dir,
            image_size=image_size,
            max_feature_count=max_feature_count,
            jobini=jobini
        )
    elif method == "patchcore":
        backbone_name = backbone if len(backbone) > 0 else "resnet18" 
        model = PatchCore(
            f_coreset=f_coreset, 
            coreset_eps=coreset_eps, 
            backbone_name=backbone_name,
            out_indices=out_indices,
            results_dir=results_dir,
            image_size=image_size,  # Examples: 320 or [640, 320]
            max_feature_count=max_feature_count,
            start_pos=start_pos,
            end_pos=end_pos,
            jobini=jobini,
            score_normalization_enabled=score_normalization.get('enabled', True),
            score_normalization_min_train_patches=score_normalization.get('min_train_patches', 4),
            score_normalization_scale_floor_quantile=score_normalization.get('scale_floor_quantile', 0.2),
            score_normalization_scale_cap_quantile=score_normalization.get('scale_cap_quantile', None),
            score_normalization_smooth_scale=score_normalization.get('smooth_scale', True),
            score_normalization_smooth_kernel=score_normalization.get('smooth_kernel', 3),
            score_normalization_threshold_quantile=score_normalization.get('threshold_quantile', 0.999),
            score_normalization_clamp_min_zero=score_normalization.get('clamp_min_zero', True),
        )
    # model = model.to(device)
    print(f"\n█│ Running {method} on {cls} dataset.")
    print(  f" ╰{'─'*(len(method)+len(cls)+23)}\n")
    train_ds, test_ds = MVTecDataset(cls,size=image_size,dataset_dir=dataset_dir,resize_method=resize_method).get_dataloaders()
    jobini.set_exec_progress(10)

    print("   Training ...")
    model.fit(train_ds)
    jobini.set_exec_progress(95)
    
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

def load_config(cfg_path):
    with codecs.open(cfg_path, 'r', 'utf-8') as f:
        cfg = yaml.load(f,Loader=yaml.FullLoader)
    return cfg

def md5_file(path):
    with open(path, 'rb') as f:
        hashval = hashlib.new('sha256', f.read()).hexdigest()
        return hashval[0:8]
        
def output_clear(dataset_dir):
    model_txt = os.path.join(dataset_dir, "model.txt")    
    if os.path.exists(model_txt):
        os.remove(model_txt)
     
def output_result(dataset_dir, result_dir, output_dir, output_name, job_no):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    src_path = os.path.join(result_dir, "patch_lib.ts")
        
    # add manifest.json / coco.json
    with zipfile.ZipFile(src_path, 'a') as zipf:
        manifest_file = os.path.join(dataset_dir, 'manifest.json')        
        if os.path.exists(manifest_file):
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                base_image_name = manifest.get('model', {}).get('base_image', '')
                base_image_file = os.path.join(dataset_dir, 'images', 'orig', base_image_name)
                assert os.path.exists(base_image_file)
                zipf.write(base_image_file, base_image_name)
            zipf.write(manifest_file, 'manifest.json')

        coco_file = os.path.join(dataset_dir, 'coco.json')
        if os.path.exists(coco_file):
            zipf.write(coco_file, 'coco.json')
            
        iaz_file = os.path.join(dataset_dir, f'{job_no}.iaz')
        if os.path.exists(iaz_file):
            zipf.write(iaz_file, f'{job_no}.iaz')

    #copy patch_lib.ts
    dst_name = output_name.replace('[md5]', md5_file(src_path))
    dst_name = dst_name.replace('.ts', '.ts2')
    
    dst_path = os.path.join(output_dir, dst_name)
    shutil.copyfile(src_path, dst_path)
    
    #写入 model.txt 文件
    model_txt = os.path.join(dataset_dir, "model.txt")
    f = open(model_txt, "w")
    f.write(dst_name)
    f.close()

'''
CHANGE 
python ./indad/run.py patchcore --dataset_dir /opt/data/private/dataset/202207/28/2207282016931 --dataset dataset --result_dir /opt/data/private/dataset/202207/28/2207282016931/result  --output_dir /opt/data/private/dataset/model/202207/28 --output_name wikad1_2207282016931_{sha}.ts
TO:
sys.executable + ' {} patchcore --cfg-path {} --output_dir {} --output_name {}'.format(train_file, datasetYml, output_dir, output_name)
'''
def get_auto_image_height(dataset_dir,dataset):
    dir1 = os.path.join(dataset_dir,dataset)
    img_dir = os.path.join(dir1,'train/good')
    img_file = os.path.join(img_dir,os.listdir(img_dir)[0])
    image_size = cv2.imread(img_file).shape
    height = image_size[0]
    if height % 32 != 0:
        height = (divmod(height,32)[0]+1)*32
    return height


@click.command()
@click.option("--cfg_tpl", default="./config/tpl/patchcore-cv2.yml", required=False,type=str,help="cfg template path")
@click.option("--cfg_path", default="", required=False,type=str,help="cfg variable path")
@click.option("--output_dir", default="./outputs", required=False,type=str,help="output dir")
@click.option("--max_feature_count", default=0, required=False, type=int, help="max feature count")
@click.option("--start_pos", default=0, required=False, type=int, help="start pos")
@click.option("--end_pos", default=0, required=False, type=int, help="end pos")
def cli_interface(cfg_tpl: str, cfg_path: str, output_dir: str, max_feature_count: int, start_pos: int, end_pos : int):
    # print(f'type(max_feature_count)={type(max_feature_count)}, max_feature_count={max_feature_count}')
    
    cfg = load_config(cfg_path)
    
    method = cfg['method']
    backbone = cfg['backbone']
    job_no = cfg['job_no']
    image_size = cfg['image_size']
    f_coreset = cfg['f_coreset']        #fraction the number of training samples
    coreset_eps = 0.90                  #sparse projection parameter
    score_normalization = cfg.get('score_normalization', {})
    feature_indices = "2,3"
    dataset_dir = cfg['dataset_dir']
    dataset = cfg['dataset']
    result_dir = cfg['result_dir']

    if isinstance(image_size, int):
        pass
    elif isinstance(image_size, list) and len(image_size) == 2:
        width, height = image_size        
        if width % 32 != 0:
            width = (divmod(width, 32)[0] + 1) * 32
        if height % 32 != 0:
            height = (divmod(height, 32)[0] + 1) * 32
            
        image_size = [width, height]
    else:
        raise ValueError(f'Invalid image_size: {image_size}')
        
    # 如果尺寸自适应,图像宽度按照高度等比例缩放
    if (isinstance(image_size, int) and image_size == 0) or (isinstance(image_size, list) and image_size == [0, 0]):
        image_size = get_auto_image_height(dataset_dir,dataset)
    print(f'cli_interface: image_size={image_size}')
    
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
    
    #缩放方法
    resize_method = "cv2"
    #特征索引
    out_indices=tuple([int(i) for i in feature_indices.split(',')])

    #输出目录
    curr_time = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
    result_name = '{0}_{1}_{2}_{3}_{4}'.format(curr_time,method,backbone,feature_indices, image_size if isinstance(image_size, int) else f'{image_size[0]}x{image_size[1]}' )
    result_dir = os.path.join(result_dir, result_name )
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    #运行模型
    total_results, run_info = run_model(jobini,method, backbone, resize_method, out_indices, dataset, dataset_dir,result_dir,image_size, f_coreset,coreset_eps, max_feature_count, start_pos, end_pos, score_normalization)
    
    #输出结果
    print_and_export_results(total_results, method, result_dir)
    
    #保存模型
    backbone_name = backbone.replace('_','-')
    fmap_size = '%dx%d' % (run_info['fmap_size'][0], run_info['fmap_size'][1])
    
    image_info = "%dx%dx%d" %(run_info['image_shape'][0],run_info['image_shape'][1],run_info['image_shape'][2])
    print(f'image_info={image_info}')
    
    manifest_file = os.path.join(dataset_dir, 'manifest.json')
    if os.path.exists(manifest_file):
        with open(manifest_file, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
            
        image_size = manifest['model']['image_size']
        image_info = "%dx%dx%d" %(image_size[0],image_size[1],image_size[2])
        print(f'Update: image_size={image_size}, image_info={image_info}')
    
    output_name = f'{method}_{job_no}_{resize_method}_{backbone_name}_23_{fmap_size}_{image_info}_fp32_[md5].ts'
    
    output_result(dataset_dir,result_dir,output_dir,output_name, job_no)   
    jobini.set_exec_progress(100)
        
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cli_interface()
