# ---------------------
# Bilibili: 随风而息
# Time：2022/5/5 19:02
# ---------------------
import argparse
import datetime
import math
import os
from concurrent.futures import ThreadPoolExecutor

import pyautogui
import win32api
import platform
import subprocess
import threading
import time
from collections import namedtuple, OrderedDict
from pathlib import Path
from subprocess import check_output
import cv2
import mss
import numpy as np
import pkg_resources as pkg
import pydirectinput
import pandas as pd
import torch
import win32con
from colorama.win32 import windll
from simple_pid.PID import PID
from torch import nn
from pynput.mouse import Button, Controller, Listener
from aim.mydata import letterbox, non_max_suppression, scale_coords, xyxy2xywh, LOGGER
from models.experimental import attempt_load
from utils.torch_utils import time_sync

VERBOSE = str(os.getenv('YOLOv5_VERBOSE', True)).lower() == 'true'  # 全局详细模式
FILE = Path(__file__).resolve()
# 文件位置
ROOT = FILE.parents[0]
# print("ROOT",FILE.parents[0])
MOUSE_STATE = False
AIM_X_Y = (0, 0)
mutex = threading.Lock()


# 查看开始时间
def date_modified(path=__file__):
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


# githuab
def git_describe(path=Path(__file__).parent):  # 路径必须是目录
    # 返回可读的 git 描述
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError:
        return ''  # 不是 git 存储库


# 查看CUDA和pytorch版本
def select_device(device='', batch_size=0, newline=True):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'YOLOv5 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    # 使用cpu
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 强制 torch.cuda.is_available() = False
    # 使用其他设备
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # 设置环境变量 - 必须在断言is_available() 之前
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"无效的CUDA'--device {device}' 请使用“--device cpu 或其他英伟达 CUDA 设备"
    cuda = not cpu and torch.cuda.is_available()
    # 查看CUDA数量
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # 设备数量
        if n > 1 and batch_size > 0:  # 检查 batch_size 是否可以被 device_count 整除
            assert batch_size % n == 0, f'batch-size {batch_size} 不是 GPU 数量的倍数 ,请使用{n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            # 查看显存大小
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2:.0f}MiB)\n"
    else:
        s += 'CPU\n'
    if not newline:
        s = s.rstrip()
    LOGGER.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')


# 打印 argparser 参数
def print_args(name, opt):
    LOGGER.info(colorstr(f'项目:{name} \n参数:') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))


# 后缀池,获取后缀函数调用
def export_formats():
    # YOLOv5 导出格式
    x = [['PyTorch', '-', '.pt'],
         ['ONNX', 'onnx', '.onnx'],
         ['TensorRT', 'engine', '.engine']]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix'])


# 检查文件是否有可接受的后缀     获取后缀函数调用
def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    if file and suffix:  # 有路径和.pt后缀
        if isinstance(suffix, str):  # 判断.pt是否str
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:  # 判断file是否元组,不是就改为列表
            s = Path(f).suffix.lower()  # 将项目路径和文件后缀拼接然后转换字符串中所有大写字符为小写
            if len(s):
                assert s in suffix, f"{msg}{f} 可接受的后缀是{suffix}"  # 如果项目路径下没有suffix后缀,返回错误
                # 判断一个表达式,为 false 的时候触发异常,条件不满足程序运行的情况下直接返回错误,程序中止
                # f:用大括号 {} 表示被替换字段，其中直接填入替换内容


# onnx调用  检查安装依赖调用
def colorstr(*input):
    # 为字符串着色 https://en.wikipedia.org/wiki/ANSI_escape_code，即 colorstr('blue', 'hello world')    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    # 颜色参数，字符串
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {'black': '\033[30m',  # 基本颜色
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # 彩色
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # 杂项
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


# onnx 检测调用 # 检查网络连接
def check_online():
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # 检查主机可访问性
        return True
    except OSError:
        return False


# onnx检查调用      返回与平台相关的表情符号安全版本的字符串
def emojis(str=''):
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


# onnx调用 用法：@try_except 装饰器
def try_except(func):
    # 尝试除外功能。 用法：@try_except 装饰器
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return handler


# onnx调用    # 检查安装的依赖项是否满足要求（通过 *.txt 文件或包列表）
@try_except
def check_requirements(requirements=ROOT / 'requirements.txt', exclude=(), install=True):
    # 设置颜色参数，字符串
    prefix = colorstr('red', 'bold', 'requirements:')
    # 判断requirements.txt 是否在为str
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        # 判断是否在路径exists:查询路径是否存在
        assert file.exists(), f"{prefix} {file.resolve()} 没找到，检查失败。"
        # 存在,则打开requirements.txt
        with file.open() as f:
            # 先判断requirements.t是否可用,False则用openvino-dev
            requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
    # 否则为列表或元组类型包
    else:

        requirements = [x for x in requirements if x not in exclude]  # 应为类型 'collections.Iterable'，但实际为 'Path'
    n = 0  # 包更新数量
    # 遍历
    for r in requirements:
        try:
            pkg.require(r)
        except Exception:  # 如果不满足要求，则 返回未找到分发 或 版本冲突
            s = f"{prefix} {r} 未找到，这是YOLOv5要求的"
            # 是否更新,默认为True
            if install:
                LOGGER.info(f"{s}, 正在尝试自动更新...")
                try:
                    # 检查互联网连接
                    assert check_online(), f"'pip install {r}' 已跳过（原因:互联网离线）"
                    # 升级
                    LOGGER.info(check_output(f"pip install '{r}'", shell=True).decode())
                    n += 1
                # 返回未知错误
                except Exception as e:
                    LOGGER.warning(f'{prefix} {e}')
            # 是否更新,False
            else:
                LOGGER.info(f'{s}. 请安装并重新运行您的命令.')

    # 如果包(库)更新完成
    if n:
        source = file.resolve() if 'file' in locals() else requirements  # locals 函数更新并以字典形式返回当前全部局部变量
        s = f"{prefix} {n} 库{'s' * (n > 1)} 每更新一次 {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', '重新启动运行时或重新运行命令以使更新生效')}\n"
        LOGGER.info(emojis(s))


# tensorrt   # 检查版本与所需版本
def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # 对比版本
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    # 如果版本满足反回True
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    # 打印
    s = f'YOLOv5 要求 {name}{minimum} , 但是当前已安装{name}{current} '  # 字符串
    if hard:  # 默认跳过
        assert result, s  # 判断满足最低要求
    if verbose and not result:  # result为True
        LOGGER.warning(s)
    return result  # 返回True


# 检查图像大小
def make_divisible(x, divisor):
    # 返回最接近的 x 可被除数整除
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


# 主函数调用,检查图像大小
def check_img_size(imgsz, s=32, floor=0):
    # 验证图像大小是每个维度中 stride 的倍数
    if isinstance(imgsz, int):  # 整数，即 img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # 列表即 img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f'WARNING: --img-size {imgsz} 必须是最大步幅的倍数 {s}, 更新到 {new_size}')
    return new_size


# 主类
class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend 类，用于在各种后端进行 python 推理
    def __init__(self, weights='yolov5s.pt', device=None):

        super().__init__()
        # 判断传入的是模型列表还是单模型,默认为单模型
        w = str(weights[0] if isinstance(weights, list) else weights)
        # 获取后缀
        pt, onnx, engine = self.model_type(w)
        # 分配默认值,但是stride后面会重新定义 默认64,无用,可以删除
        stride, names = 64, [f'class{i}' for i in range(1000)]
        # PyTorch 框架
        if pt:
            # 加载pt神经网络
            model = attempt_load(weights if isinstance(weights, list) else w, map_location=device)
            # 设置模型步幅,重新定义stride为32
            stride = max(int(model.stride.max()), 32)
            # 获取类名
            names = model.module.names if hasattr(model, 'module') else model.names  # hasattr:判断对象是否包含对应的属性
            # 为 to()、cpu()、cuda()、half()   显式赋值
            self.model = model

        # onnxRuntime 框架
        elif onnx:
            # 宏声明
            LOGGER.info(f'正在为{w}加载onnxruntime...')
            # 产看cuda是否可用    False or True
            cuda = torch.cuda.is_available()
            # 选择使用onnxruntime-gpu或者onnxruntime
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            # 导入
            import onnxruntime
            # 如果cuda可用,执行CUDA或者,cpu,若cuda不可用,使用cpu执行
            providers = ['CUDA执行', 'CPU执行'] if cuda else ['CPU执行']
            # 加载onnx模型 ,又到了熟悉的一幕
            session = onnxruntime.InferenceSession(w)

        # TensorRT ,顾名思义,最想要的
        elif engine:
            # 宏声明
            LOGGER.info(f'正在为{w}加载TensorRT...')
            # 导入tensorrt
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            # 要求TesoroRt >=7.0.0
            check_version(trt.__version__, '7.0.0', hard=True)
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))  # namedtuple是继承自tuple的子类。
            # 写入日志
            logger = trt.Logger(trt.Logger.INFO)
            # 打开引擎,写入模型
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                # 加载模型
                model = runtime.deserialize_cuda_engine(f.read())
            bindings = OrderedDict()  # 对字典对象中元素的排序
            # 遍历模型????不懂
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = tuple(model.get_binding_shape(index))
                data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
                bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            context = model.create_execution_context()
            batch_size = bindings['images'].shape[0]
        self.__dict__.update(locals())  # 将所有变量分配给 self

    '''
    推理
    '''

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 多后端推理
        global y
        b, ch, h, w = im.shape  # batch, channel, height, width
        # print(im.shape)
        # PyTorch
        if self.pt:
            # 这里才是真正的推理
            y = self.model(im, augment=augment, visualize=visualize)
            return y

        # ONNX Runtime
        elif self.onnx:
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]

        # TensorRT
        elif self.engine:
            assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = self.bindings['output'].data
        y = torch.tensor(y)  # if isinstance(y, np.ndarray) else y
        return (y, []) if val else y

    # 获取后缀的静态方法
    @staticmethod
    def model_type(p='path/to/model.pt'):
        # 从模型路径返回模型类型，即 path='path/to/model.onnx' -> type=onnx
        suffixes = list(export_formats().Suffix)  # 导出后缀
        check_suffix(p, suffixes)  # 传入路径,和后缀读取
        print(f'支持此后缀模型:{suffixes}')
        p = Path(p).name  # 消除跟随分隔符
        pt, onnx, engine = (s in p for s in suffixes)  # 遍历suffixes后缀池,如果在后缀池有对应的后缀就赋值
        return pt, onnx, engine


# mss截图

scr = mss.mss()


def grab_screen_mss(monitor):
    return cv2.cvtColor(np.array(scr.grab(monitor)), cv2.COLOR_BGRA2BGR)


def run(weights=ROOT / 'yolov5s.pt',  # 权重路径

        conf_thres=0.65,  # 置信度
        imgsz=(640, 640),  # inference size (height, width)
        iou_thres=0.45,  # NMS IOU阈值
        max_det=1000,  # 每张图像的最大检测次数
        device='',  # 设备选择,即 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # 按类别过滤：--class 0，或--class 0 2 3
        half=False,  # 使用 FP16 半精度推理
        img_size=(640, 640),  # 输入图片尺寸
        region=(1.0, 1.0),  # 截图大小
        resize_window=1,

        ):
    # 引入 鼠标状态
    global MOUSE_STATE
    '''加载模型'''
    # 设置设备
    device = select_device(device)  # select_device:cuda或者cpu的信息,如显卡名称,显卡索引等
    # 根据模型类别加载引擎
    weights = str(weights)  # 将路径转为字符串,也可以直接用路径,看警告不爽而已
    model = DetectMultiBackend(weights, device=device)
    # 赋值
    stride, names, pt, onnx, engine = model.stride, model.names, model.pt, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)
    '''half'''
    # 半精度设置
    half &= (pt or onnx or engine) and device.type != 'cpu'  # 只有CUDA才支持半精度
    # pt精度
    if pt:
        model.model.half() if half else model.model.float()  # cpu 为单精度,cuda为半精度

    '''加载图片'''
    # 截图设置

    top_x, top_y, x, y = 0, 0, 1920, 1080  # x,y 屏幕大小,top是原点
    img0_x, img0_y = int(x * region[0]), int(y * region[1])  # 截图的宽高
    top_x, top_y = int(top_x + x // 2 * (1. - region[0])), int(top_y + y // 2 * (1. - region[1]))  # 截图区域的原点
    monitor = {'left': top_x, 'top': top_y, 'width': img0_x, 'height': img0_y}

    # 截图
    t0 = time.time()
    cv2.namedWindow('小鹿', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('小鹿', int(img0_x * resize_window), int(img0_y * resize_window))
    # pid_x = PID(1, 0.0, 0.00, setpoint=0, sample_time=0.01, output_limits=(-100,100))
    # pid_y = PID(1, 0.2, 1, setpoint=0, sample_time=0.01, output_limits=(-100,100))
    fps = 0
    while True:
        if not cv2.getWindowProperty('小鹿', cv2.WND_PROP_VISIBLE):
            cv2.destroyAllWindows()
            exit('程序结束...')
            break
        img0 = grab_screen_mss(monitor)
        img0 = cv2.resize(img0, (img0_x, img0_y))

        # 加载单张图片
        # img_path = 'imags_14.jpg'
        # img0 = cv2.imread(img_path)
        # img_hw = img0.shape[:2]
        # print(img_hw)

        '''预处理'''
        img = letterbox(img0, img_size, stride=stride, auto=False)[0]
        # 转tensor
        img = img.transpose((2, 0, 1))[::-1]  # HWC 转 CHW，BGR 转 RGB
        img = np.ascontiguousarray(img)
        # 放入设备
        img = torch.from_numpy(img).to(device)  #
        # uint8 转 fp16/32
        img = img.half() if half else img.float()
        # 归一化
        img /= 255
        # 扩大批量调暗
        if len(img.shape):
            img = img[None]

        t2 = time_sync()

        '''推理'''
        if "pt" in weights:
            pred = model(img, augment=False, visualize=False)[0]
        else:
            pred = model(img, augment=False, visualize=False)

        t3 = time_sync()
        '''后处理'''
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 过程预测
        aims = []
        for i, det in enumerate(pred):  # 每张图片
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    # bbox:(tag, x_center, y_center, x_width, y_width)
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 得到归一化xywh
                    line = (cls, *xywh)  # 将标签和坐标一起写入
                    aim = ('%g ' * len(line)).rstrip() % line
                    aim = aim.split(' ')
                    # time.sleep(5)
                    aims.append(aim)
                    # print(aims)
            if len(aims):

                # 绘制方框
                for i, det in enumerate(aims):
                    _, x_center, y_center, width, height = det  # 将det里的数据分装到前面   rc_x,y  表示归化后的比例坐标
                    x_center, width = img0_x * float(x_center), img0_x * float(width)  # (776, 1376)
                    y_center, height = img0_y * float(y_center), img0_y * float(height)
                    top_left = (int(x_center - width / 2.), int(y_center - height / 2.))
                    bottom_right = (int(x_center + width / 2.), int(y_center + height / 2.))
                    color = (0, 255, 0)  # RGB     框的颜色

                    # cv2.putText(img0, names[i], top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (34, 150, 102), 2)  # 添加标签名
                    # cv2.putText(img0, names[i], top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)  # 添加标签名
                    cv2.rectangle(img0, top_left, bottom_right, color, thickness=2)  # thickness代表线条粗细
                    # x y 位置
                    location_x = round((top_left[0] + bottom_right[0]) / 2 + top_x)
                    location_y = round(top_left[1] + (bottom_right[1] - top_left[1]) / 8 + top_y)
                    # 移到鼠标
                    if MOUSE_STATE and len(aims) > 0:
                        # 调用pid自瞄
                        auto_aim(location_x, location_y, int(width), i)
                        break

                    # 截图保存
                    # if len(aims) > 1 and fps == 60:
                    #     monitor1 = {'left': 427, 'top': 103, 'width': 1143, 'height': 872}
                    #     frame = grab_screen_mss(monitor1)
                    #     print("需要保存")
                    #     cv2.imwrite("D:/Downloads/images/" + str(round(time.time())) + ".jpg", frame)
                    #     # cv2.imwrite('savedImage.jpg', frame)
                    #     cv2.waitKey(1)
                    #     fps = 0
                    # if not fps == 60:
                    #     fps = fps + 1

                # 绘制fps
        cv2.putText(img0, "FPS:{:.1f}".format(1. / (time.time() - t0)), (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 0, 255), 3)

        # 绘制推理时间
        cv2.putText(img0, "time:({:.3f}s)".format(t3 - t2), (0, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (34, 150, 102), 3)
        # 打印推理时间
        # LOGGER.info("推理：({:.3f}s)".format(t3 - t2),)

        t0 = time.time()
        # 2716490456
        cv2.imshow('小鹿', img0)
        cv2.waitKey(1)


def auto_aim(x, y, width, i, ):
    global AIM_X_Y
    coordinate = win32api.GetCursorPos()
    coordinate = (960, 540)
    # 设置瞄准范围
    # 圆形范围
    result = math.sqrt(
        math.pow(
            coordinate[0] -
            x,
            2) +
        math.pow(
            coordinate[1] -
            y,
            2))
    if result < width * 6:
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 960, 540, 0, 0)
        AIM_X_Y = (x, y)
        print("类型", i, "定位坐标", x, ",", y, "AIM_X_Y", AIM_X_Y)


# PID瞄准
def pid_aim():
    global MOUSE_STATE, AIM_X_Y
    print("PID线程启动")
    pid_x = PID(0.02, 0.00, 0.00, sample_time=0, output_limits=(-100, 100))  # 0.006
    pid_y = PID(0.02, 0.00, 0.00, sample_time=0, output_limits=(-100, 100))
    # while True:
    #     aim_start = time.time()
    #     if MOUSE_STATE and AIM_X_Y[0] != 0 and AIM_X_Y[1] != 0:
    #         coordinate = win32api.GetCursorPos()
    #         pid_x.setpoint = AIM_X_Y[0]
    #         pid_y.setpoint = AIM_X_Y[1]
    #         x = pid_x(coordinate[0])
    #         y = pid_y(coordinate[1])
    #         print("需求", AIM_X_Y, "pid线程移动到：", x, y)
    #         win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, math.ceil(x), math.ceil(y), 0, 0)
    while True:
        aim_start = time.time()
        if MOUSE_STATE and AIM_X_Y[0] != 0 and AIM_X_Y[1] != 0:
            # 移动到屏幕中心
            # win32api.SetCursorPos((960, 540))
            # 鼠标左键按下
            # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            pid_x.setpoint = AIM_X_Y[0]
            pid_y.setpoint = AIM_X_Y[1]
            x, y = win32api.GetCursorPos()
            x = pid_x(960)
            y = pid_y(540)
            # if x > 0:
            #     x = math.ceil(x)
            # else:
            #     x = math.floor(x)
            # if y > 0:
            #     y = math.ceil(y)
            # else:
            #     y = math.floor(y)
            print("需求", AIM_X_Y, "pid线程移动到：", x, y)
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, round(x), round(y), 0, 0)
            # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            if not MOUSE_STATE or (abs(x) < 1 and abs(y) < 1):
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 960, 540, 0, 0)
                win32api.SetCursorPos((960, 540))
                AIM_X_Y = (0, 0)
        else:
            AIM_X_Y = (0, 0)
        sellp_time = (1 - (time.time() - aim_start) * 60) / 60
        if sellp_time > 0:
            time.sleep(sellp_time)
        else:
            time.sleep(0.02)


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'MODEL/yolov5s.onnx',help='模型路径,支持pt,onnx,tensorrt')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'data/best.pt',
                        # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'data/yolov5s.pt',
                        help='模型路径,支持pt,onnx,tensorrt')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[416], help='inference size h,w')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度')
    parser.add_argument('--conf-thres', type=float, default=0.55, help='置信度')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU阈值')
    parser.add_argument('--max-det', type=int, default=1000, help='每张图像的最大检测次数')
    parser.add_argument('--device', default='', help='设备选择, 0或者cpu,空为自动选择设备,cuda优先')
    parser.add_argument('--classes', nargs='+', type=int, help='按类别过滤：--class 0，或--class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='与类别无关的 NMS')
    parser.add_argument('--half', action='store_true', help='使用 FP16 半精度推理')

    # parser.add_argument('--region', type=tuple, default=(1, 1), help='检测范围；分别为横向和竖向，(1.0, 1.0)表示全屏检测，')
    parser.add_argument('--region', type=tuple, default=(0.3, 0.3), help='检测范围；分别为横向和竖向，(1.0, 1.0)表示全屏检测，')
    parser.add_argument('--resize-window', type=float, default=1 / 2, help='缩放实时检测窗口大小')
    opt = parser.parse_args()
    # print_args(FILE.stem, opt)
    return opt


def on_click(x, y, button, pressed):
    global MOUSE_STATE, AIM_X_Y
    mutex.acquire()
    # MOUSE_STATE = pressed
    # MOUSE_STATE = True

    if button == Button.right and pressed:
        MOUSE_STATE = not MOUSE_STATE
    mutex.release()
    print('{0} at {1}'.format('按下Pressed' if pressed else '松开Released', (x, y)), button, MOUSE_STATE)
    if not pressed:
        AIM_X_Y = (0, 0)


def click_main():
    with Listener(on_click=on_click) as listener:
        listener.join()


def reset():
    global MOUSE_STATE
    while True:
        if MOUSE_STATE:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            # 移动到屏幕中心
            win32api.SetCursorPos((960, 540))
            # 鼠标左键按下
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(1)


# 检查是否为管理员权限
def is_admin():
    try:
        return windll.shell32.IsUserAnAdmin()
    except OSError as err:
        print('OS error: {0}'.format(err))
        return False


def main(opt):
    pool = ThreadPoolExecutor(max_workers=3)
    check_requirements(exclude=('tensorboard', 'thop'))
    if not is_admin():
        print("请以管理员身份运行")
        return
    # pool.submit(click_main)
    # pool.submit(run, **vars(opt))
    # print(pool.submit(pid_aim).result())
    # print(pool.submit(run, **vars(opt)).result())

    try:
        pool.submit(click_main)
        # pool.submit(reset)
        pool.submit(run, **vars(opt))
        print(pool.submit(pid_aim).result())
    except Exception as e:
        print('yolo出现异常:', e)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
