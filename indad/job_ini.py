import configparser
import os
import time

class JobIni:
    def __init__(self, dataset_dir, file_name=None):
        if dataset_dir.endswith('_s1'):           # 二阶段训练时使用总的job.ini文件
            dataset_dir = dataset_dir[:-3]
        self.dataset_dir = dataset_dir
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        if file_name:
            self.ini_path = os.path.join(dataset_dir, file_name)
        else:
            self.ini_path = os.path.join(dataset_dir,'job.ini')
        
    def init_file(self):
        if not self.is_valid():
            if os.path.exists(self.ini_path):
                os.remove(self.ini_path)
        config = configparser.ConfigParser()
        config.read(self.ini_path, encoding="utf8")
        if not config.has_section("job"):
            config.add_section("job")
        with open(self.ini_path,"w",encoding="utf-8") as f:
            config.write(f)

    def is_valid(self):
        try:
            config = configparser.ConfigParser()
            config.read(self.ini_path, encoding="utf8")
            if not config.has_section("job"):
                config.add_section("job")
            with open(self.ini_path,"w",encoding="utf-8") as f:
                config.write(f)
            config.set("job", 'jobtime', str(time.time()))
            return True
        except:
            return False            
            
    def get(self,key,default = None):
        config = configparser.ConfigParser()
        config.read(self.ini_path, encoding="utf8")
        if not config.has_option("job", key):
            return default
        return config.get("job", key)
        
    def set(self,key,value):
        config = configparser.ConfigParser()
        config.read(self.ini_path, encoding="utf8")
        if isinstance(value,int):
            config.set("job", key, str(value))
        elif isinstance(value,str):
            config.set("job", key, value)
        else:
            config.set("job", key, str(value))
        with open(self.ini_path,"w",encoding="utf-8") as f:
            config.write(f)
    
    def set_total_stages(self, total_stages):
        self.set(f'total_stages', total_stages)

    def get_total_stages(self):
        value = self.get(f'total_stages')
        return int(value)

    def get_exec_status(self):
        return self.get(f'exec_status')
    
    def set_exec_status(self, status, stage=1, last_stage=False):
        print('set_exec_status', status, stage)
        
        if last_stage:
            self.set(f'exec_status', status)
            return

        total_stages = self.get_total_stages()        
        if total_stages == 1:
            self.set(f'exec_status_{stage - 1}', status)
            self.set(f'exec_status', status)
        elif total_stages == 2:
            self.set(f'exec_status_{stage - 1}', status)
            
            if stage == 1:
                if status == 'succeeded':
                    self.set(f'exec_status', 'running')
                else: # status == 'running' or status == 'failed' :
                    self.set(f'exec_status', status)
            elif stage == 2:
                self.set(f'exec_status', status)
            else:
                raise ValueError(f'Invalid stage: {stage}')
        else:
            raise ValueError(f'Invalid total stages: {total_stages}')
    
    def get_exec_progress(self, stage=1):
        return self.get(f'exec_progress')
        
    def set_exec_progress(self, progress, stage=1):
        total_stages = self.get_total_stages()
        
        if total_stages == 1:
            self.set(f'exec_progress_{stage - 1}', progress)
            self.set(f'exec_progress', progress)
        elif total_stages == 2:
            self.set(f'exec_progress_{stage - 1}', progress)
            
            if stage == 1:
                self.set(f'exec_progress', f'{float(progress) * 0.3:.2f}')
            elif stage == 2:
                self.set(f'exec_progress', f'{30 + float(progress) * 0.7:.2f}')
            else:
                raise ValueError(f'Invalid stage: {stage}')
        else:
            raise ValueError(f'Invalid total stages: {total_stages}')

    def get_dataset_prepared(self):
        value = self.get('dataset_prepared', 0)
        return int(value)
        
    def set_dataset_prepared(self, value):
        self.set('dataset_prepared', 1)

    def get_dataset_dir(self):
        return self.get('dataset_dir', '')   
        
    def set_dataset_dir(self,value):
        self.set('dataset_dir', value)
		
if __name__ == "__main__":
    ini = JobIni('./')
    ini.clear()
    ini.set("state",1)
    ini.set("msg",'hello')