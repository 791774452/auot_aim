# # coding:utf-8
# import time
#
# from appium import webdriver
# import math
#
# from selenium.webdriver import ActionChains
#
# desired_caps = {
#     'platformName': 'Android',  # 被测手机是安卓
#     'platformVersion': '11',  # 手机安卓版本
#     'deviceName': 'xxx',  # 设备名，安卓手机可以随意填写
#     # 'appPackage': 'com.taobao.taobao',  # 启动APP Package名称
#     # 'appActivity': 'com.taobao.tao.welcome.Welcome',  # 启动Activity名称
#     'unicodeKeyboard': True,  # 使用自带输入法，输入中文时填True
#     'resetKeyboard': True,  # 执行完程序恢复原来输入法
#     'noReset': True,  # 不要重置App，如果为False的话，执行完脚本后，app的数据会清空，比如你原本登录了，执行完脚本后就退出登录了
#     'newCommandTimeout': 6000,
#     'automationName': 'UiAutomator2'
#
# }
# driver = webdriver.Remote('http://127.0.0.1:4723/wd/hub', desired_caps)
#
# # 获取屏幕的size
# size = driver.get_window_size()
# print(size)
# # 屏幕宽度width
# print(size['width'])
# # 屏幕高度width
# print(size['height'])
#
#
# def swipeUp(driver, t=500, n=1):
#     '''向上滑动屏幕'''
#     l = driver.get_window_size()
#     x1 = l['width'] * 0.5  # x坐标
#     y1 = l['height'] * 0.75  # 起始y坐标
#     y2 = l['height'] * 0.25  # 终点y坐标
#     for i in range(n):
#         driver.swipe(x1, y1, x1, y2, t)
#
#
# def swipeDown(driver, t=500, n=1):
#     '''向下滑动屏幕'''
#     l = driver.get_window_size()
#     x1 = l['width'] * 0.5  # x坐标
#     y1 = l['height'] * 0.25  # 起始y坐标
#     y2 = l['height'] * 0.75  # 终点y坐标
#     for i in range(n):
#         driver.swipe(x1, y1, x1, y2, t)
#
#
# def swipLeft(driver, t=500, n=1):
#     '''向左滑动屏幕'''
#     l = driver.get_window_size()
#     x1 = l['width'] * 0.75
#     y1 = l['height'] * 0.5
#     x2 = l['width'] * 0.25
#     for i in range(n):
#         driver.swipe(x1, y1, x2, y1, t)
#
#
# def swipRight(driver, t=500, n=1):
#     '''向右滑动屏幕'''
#     l = driver.get_window_size()
#     x1 = l['width'] * 0.25
#     y1 = l['height'] * 0.5
#     x2 = l['width'] * 0.75
#     for i in range(n):
#         driver.swipe(x1, y1, x2, y1, t)
#
#
# if __name__ == "__main__":
#     a_b = driver.get_window_size()
#     a, b = a_b['width'] / 2, a_b['height'] / 2  # 圆点坐标
#
#     w = 200  # 圆平均分为10份
#     m = (2 * math.pi) / w  # 一个圆分成10份，每一份弧度为 m
#
#     r = 500  # 半径
#
#     point_list = ""
#
#     # for i in range(0, w + 1):
#     #     x = a + r * math.sin(m * i)
#     #     y = b + r * math.cos(m * i)
#     #     point_list += " {},{}".format(x, y)
#     #     driver.swipe(x, y, x, y, 50)
#     sta = time.time()
#     while time.time() - sta < 10:
#         actions = ActionChains(driver)
#         # 使用click_and_hold方法按住元素并保持
#         actions.w3c_actions.pointer_action.click()
#         actions.w3c_actions.pointer_action.move_to_location(200,200)
#
#         # 最后使用perform方法执行以上操作。
#         actions.perform()
#     print(a_b)
#
#     # print(point_list)
#     # sleep(1)
#     # swipLeft(driver, n=1)
#     # sleep(1)
#     # swipRight(driver, n=1)
#     # sleep(1)
#     # swipeDown(driver, n=1)
#     # sleep(1)
#     # swipeUp(driver, n=1)
#     # driver.quit()
#
# # 退出程序，记得之前没敲这段报了一个错误 Error: socket hang up 啥啥啥的忘记了，有兴趣可以try one try
# driver.quit()
import subprocess


def adb(command):
    proc = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE, shell=True)
    (out, _) = proc.communicate()
    return out.decode('utf-8')
adb('adb shell sendevent /dev/input/event7 0001 0330 00000001')
adb('adb shell sendevent /dev/input/event7 0003 0058 00000001')
adb('adb shell sendevent /dev/input/event7 0003 0053 00000290')
adb('adb shell sendevent /dev/input/event7 0003 0054 00000469')
adb('adb shell sendevent /dev/input/event7 0000 0002 00000000')
adb('adb shell sendevent /dev/input/event7 0000 0000 00000000')
