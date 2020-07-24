from subprocess import getoutput
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from pandas.io.clipboard import clipboard_get
from sklearn.metrics import accuracy_score,matthews_corrcoef,r2_score,mean_squared_error,mean_absolute_error,confusion_matrix
from pathlib import Path
import time, random, string, tempfile

def calc_entropy(x):
    """
    calculate scaled shanno entropy of x
    
    parameter:
    -----
    x : 1D-np.ndarray or dict
    
    eg:
    ----
        calc_ent( np.array([1,1,2]) )
        calc_ent( {1:2, 2:1} )
    """
    if type(x) == np.ndarray:
        length = x.shape[0]
        x_value_list = set(x)
        ent = 0.0
        for x_value in x_value_list:
            p = (x == x_value).sum() / length
            logp = np.log2(p)
            ent -= p * logp
    elif type(x)==dict:
        x_count_list = x.values()
        length = sum(x_count_list)
        ent = 0.0
        for counts in x_count_list:
            p = counts/length
            logp = np.log2(p)
            ent -= p * logp
    else:
        raise Exception("The type of x must be `np.ndarray` or `dict`")
    return ent/np.log2(length)
    
def generate_random_str(prefix="",suffix="",length=6)->str:
    "生成随机字符串，可以指定前缀和后缀，指定长度"
    salt = ''.join(random.sample(string.ascii_letters + string.digits, length))
    return prefix+salt+suffix

def get_temp_dir(prefix="",suffix="")->"pathlib.Path":
    "获取并创建，带有时间戳的数据储存的临时目录"
    parent_path = Path(tempfile.gettempdir())   # 获取临时目录
    time_stamp = time.strftime("%Y%m%d.%H%M%S_") # 时间戳
    dir_name = generate_random_str(prefix+time_stamp)
    temp_dir = parent_path/dir_name # 临时保存数据用的文件夹
    if temp_dir.exists():  # 如果存在，就报错
        raise Exception("A folder with the same name exists, please try again immediately.")
    else:  # 如果不存在就新建
        temp_dir.mkdir()  # 创建保存数据用的文件夹
        # pass
    return temp_dir
    
def ReadChemPropGridVerbose(path:"需要读取的寻优结果文件路径",iters:"int, 寻优的参数个数"=100)->"2d_df":
    """读取ChemProp的的寻优结果"""
    with open(path,"r") as f:
        content = f.readlines()
    r_grid = []
    for i in range(iters):
        param = content[i*3].replace("\n","")
        num_param = int(content[i*3+1].split(": ")[-1].replace("\n","").replace(",",""))
        metrics_ = float(content[i*3+2][:-14])
        r_grid.append([metrics_,num_param,param])
    metrics_name = content[2].split(" ")[-1].replace("\n","")
    r_grid_df = pd.DataFrame(r_grid,columns=[metrics_name,"num_param","param"]).sort_values(metrics_name)
    return r_grid_df
    
def Read_xls(path,sheet_number=1):
    """
    读取指定sheet的全部数据
    eg:
    data_tuple = Read_xls(path,sheet_number=1)
    data =pd.DataFrame.from_records(data_tuple[1:],columns=data_tuple[0])
    
    parameters
    -------
    path: 打开文件的路径
    sheet_number: int，读取第几个sheet的数据，first的数字为1
    """
    from win32com.client import Dispatch #系统内置包
    xlApp = Dispatch('Excel.Application')
    xlApp.Visible = True 
    xlApp.DisplayAlerts = False
    xlBook = xlApp.Workbooks.Open(path)
    sht = xlBook.Worksheets(sheet_number)
    data_tuple = sht.Range(sht.Cells(1, 1), sht.Cells(sht.UsedRange.Rows.Count, sht.UsedRange.Columns.Count)).Value
    return data_tuple

def metrics_mergeTrTe(tr_y_true, tr_y_pred, te_y_true, te_y_pred, data_type):
    """
    parameter
    -----
    tr_y_true, tr_y_pred, te_y_true, te_y_pred: 1D-np.array
    data_type: str, 'R' or 'C'
    
    return
    -----
    return: dict
    """
    if data_type == "C":
        tr = metrics_C(tr_y_true, tr_y_pred, prefix = "tr_")
        te = metrics_C(te_y_true, te_y_pred, prefix = "te_")
    elif data_type == "R":
        tr = metrics_R(tr_y_true, tr_y_pred, prefix = "tr_")
        te = metrics_R(te_y_true, te_y_pred, prefix = "te_")
    else:
        raise ValueError("data_type = 'R' or 'C'")
    tr.update(te)#合并两个字典
    return tr #返回合并后的字典
        
def metrics_C(y_true, y_pred, prefix = ""):
    """
    parameter
    -----
    y_true/y_pred: 1D-np.array
    prefix: str, return时返回字典的键前缀
    
    return
    -----
    return: dict
    """
    result = {}
    result[prefix+"TN"],result[prefix+"FP"],result[prefix+"FN"],result[prefix+"TP"] = confusion_matrix(y_true,y_pred).ravel()
    result[prefix+"SE"]  = result[prefix+"TP"] / (result[prefix+"TP"] + result[prefix+"FN"])
    result[prefix+"SP"]  = result[prefix+"TN"] / (result[prefix+"TN"] + result[prefix+"FP"])
    result[prefix+"ACC"] = accuracy_score(y_true, y_pred)
    result[prefix+"MCC"] = matthews_corrcoef(y_true, y_pred)
    return result
def metrics_R(y_true, y_pred, prefix = ""):
    """
    parameter
    -----
    y_true/y_pred: 1D-np.array   
    prefix: str, return时返回字典的键前缀
    
    return
    -----
    return: dict
    """
    result = {}
    result[prefix+"MSE"] = mean_squared_error(y_true, y_pred)
    result[prefix+"RMSE"] = np.sqrt(result[prefix+"MSE"])
    result[prefix+"MAE"] = mean_absolute_error(y_true, y_pred)
    result[prefix+"r2"] = r2_score(y_true, y_pred)
    return result

def get_clipboard():
    pass
    return clipboard_get()
    
    
def dictValue2Int(dic:dict):
    """
    键字典里面用浮点型整数值转换为整数，其余不符合要求的不转换
    eg: {"a":12.0, "b":13.8, "c":plt} -> {"a":12, "b":13.8, "c":plt}
    
    return: dict
    """
    dic_int = {}
    for k,v in dic.items():
        try:
            v_int = int(v)
            if v_int==v:
                dic_int[k] = v_int
            else:
                dic_int[k] = v
        except:
            dic_int[k] = v
    return dic_int


def splitList(X,n):
    """
    将一个1D列表打乱后等份额n拆分，返回拆分后的列表，列表的每一个元素也是列表
    eg:
       splitList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3)
    -> [[5, 4, 7], [8, 1, 10], [9, 2, 3, 6]]
    
    parameters
    -----
    X: 1D-array, of list
    n: int
    
    return
    -----
    list
    """
    out = []
    for i in range(n-1):
        first, X = train_test_split(X, train_size=1/(n-i), random_state=42)
        out.append(first)
    out.append(X)
    return out

def Sec2Time(seconds):  # convert seconds to time
    """
    convert seconds to time, eg: 00h:00m:00s
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return ("{:02d}h:{:02d}m:{:02d}s".format(h, m, s))
    
    
def _GetAvailableGpuId(remain=2, AvailableGpuStandard=50,AvailableMemoryStandard=50,num_gpu=8):
    """
    得到闲置GPU的id（已除去预留数）,利用的linux里面的`nvidia-smi`命令，利用三次取平均
    
    parameters
    -----
    remain: int,预留不使用的GPU核心数
    AvailableGpuStandard：int or float，Gpu核心利用率大于AvailableGpuStandard才算作可用Gpu
    AvailableMemoryStandard：同上，Memory可用标准
    num_gpu: int, gpu的核心数目
    
    return
    -----
    1D-np.Array, eg: array([0, 1, 2])
    """
    def _getInfo():
        GPU_Utilization = getoutput("nvidia-smi -q -d Utilization | grep -A1 Gpu | grep -o '[0-9]\+'")#
        info = np.array([int(i) for i in GPU_Utilization.split('\n')]).reshape(num_gpu,2)#第一列gpu，第二列Memory
        boolIndex = (info[:,0]<(100-AvailableGpuStandard))&(info[:,1]<(100-AvailableMemoryStandard))
        return boolIndex
    b1 = _getInfo()
    time.sleep(0.3+np.random.rand(1)[0])
    b2 = _getInfo()
    time.sleep(0.3+np.random.rand(1)[0])
    b3 = _getInfo()
    AvailableID = np.arange(0,num_gpu)[b1&b2&b3]
    return AvailableID[remain:]
    
def GetAvailableGpuId(remain=2, AvailableGpuStandard=50,AvailableMemoryStandard=50,num_gpu=8):
    """
    得到闲置GPU的id（已除去预留数）,利用的linux里面的`nvidia-smi`命令，执行一次，得到16s左右的平均状态
    
    parameters
    -----
    remain: int,预留不使用的GPU核心数
    AvailableGpuStandard：int or float，Gpu核心利用率大于AvailableGpuStandard才算作可用Gpu
    AvailableMemoryStandard：同上，Memory可用标准
    num_gpu: int, gpu的核心数目
    
    return
    -----
    1D-np.Array, eg: array([0, 1, 2])
    """
    GPU_Utilization = getoutput("nvidia-smi -q -d Utilization | grep -A1 Gpu | grep -o '[0-9]\+'")#
    info = np.array([int(i) for i in GPU_Utilization.split('\n')]).reshape(num_gpu,2)#第一列gpu，第二列Memory
    boolIndex = (info[:,0]<(100-AvailableGpuStandard))&(info[:,1]<(100-AvailableMemoryStandard))
    AvailableID = np.arange(0,num_gpu)[boolIndex]
    return AvailableID[remain:]
    
if __name__ == '__main__':
    pass