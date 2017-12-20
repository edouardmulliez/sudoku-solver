# -*- coding: utf-8 -*-

'''

Take a picture of a sudoku grid. Print an image of the solved grid below.

'''

# Uncomment these lines to see all the messages
# from kivy.logger import Logger
# import logging
# Logger.setLevel(logging.TRACE)

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.image import Image

from SudokuReader import read_solve_save

Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
    Camera:
        id: camera
        resolution: (640, 480)
        play: False
    Image:
        id: img
        source: 'crop.png'
    ToggleButton:
        text: 'Play'
        on_press: camera.play = not camera.play
        size_hint_y: None
        height: '48dp'
    Button:
        text: 'Capture'
        size_hint_y: None
        height: '48dp'
        on_press: root.capture()
''')


class CameraClick(BoxLayout):
    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        camera = self.ids['camera']
        camera.export_to_png("input.png")
        print("Captured")
        read_solve_save("input.png", "output.png")
        img = self.ids['img']
        img.source = "output.png"
        img.reload()


class TestCamera(App):

    def build(self):
        return CameraClick()


TestCamera().run()