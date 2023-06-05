import time

import win32api
import win32con
time.sleep(10)
win32api.SetCursorPos((960, 540))
time.sleep(3)
win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 100, 100, 0, 0)
win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

