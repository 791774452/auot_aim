import cv2
import os

#此删除文件夹内容的函数来源于网上
def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

def video_to_images(fps,path,order):
  cv = cv2.VideoCapture(path)
  if(not cv.isOpened()):
    print("\n打开视频失败！请检查视频路径是否正确\n")
    exit(0)
  # if not os.path.exists("images/"):
  #   os.mkdir("images/") # 创建文件夹
  # else:
  #   del_file('images/') # 清空文件夹
  order = order   #序号
  h = 0
  while True:
    h=h+1
    rval, frame = cv.read()
    if h == fps:
      h = 0
      order = order + 1
      if rval:
        cv2.imwrite("D:\\images\\" + str(order) + '.jpg', frame)
        cv2.waitKey(1)
      else:
        break
  cv.release()
  print(order)
  print('\nsave success!\n')
  return order

# 参数设置
fps = 60   # 隔多少帧取一张图  1表示全部取
if __name__ == '__main__':
  # video_to_images(fps,path)
  # 会在代码的当前文件夹下 生成images文件夹 用于保存图片
  # 如果有images文件夹，会清空文件夹！

  g = os.walk(r"E:\IDM\微信下载\WeChat Files\wxid_5l345992kkqs22\FileStorage\File\2022-05\UZI游戏解说\2022.04-17")
  order=2262
  for path, dir_list, file_list in g:
      for file_name in file_list:
          print(os.path.join(path, file_name))
          order=video_to_images(fps, os.path.join(path, file_name),order)