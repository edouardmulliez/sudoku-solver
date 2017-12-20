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
        play: True
    Image:
        id: img
        source: 'crop.png'
        size_hint_y: None
        height: '0dp'
    Label:
        id: lbl
        text: ''
        size_hint_y: None
        height: '48dp'
    Button:
        id: btn
        text: 'Read and solve sudoku'
        size_hint_y: None
        height: '48dp'
        on_press: root.button_action()
''')

messages = dict({0: 'Read and solved',
                 1: 'Could not find grid borders.',
                 2: 'Could not solve sudoku. \n'+
                 'Most probable issue is that the digits were not recognised correctly.'})

def show(w):
    w.size_hint_y = 1

def hide(w):
    w.size_hint_y = None
    w.height = '0dp'
    
    
class CameraClick(BoxLayout):
    def button_action(self):
        camera = self.ids['camera']
        if camera.size_hint_y is not None:
            self.capture()
        else:
            self.show_camera()
    
    def capture(self):
        '''
        Function to capture image, solve sudoku and display results
        Also update label and button texts.
        '''
        camera = self.ids['camera']
        camera.export_to_png('input.png')
        status = read_solve_save('input.png', 'output.png')
        
        # Update label
        lbl = self.ids['lbl']
        lbl.text = messages[status]
        
        # Update and display image
        img = self.ids['img']
        img.source = 'output.png'
        img.reload()
        show(img)
        hide(camera)
        
        # Change button text
        btn = self.ids['btn']
        btn.text = 'Retry'
    
    def show_camera(self):
        """
        Show camera, hide image.
        Also update label and button texts.
        """
        camera = self.ids['camera']
        img = self.ids['img']
        show(camera)
        hide(img)
        
        # Change button text
        btn = self.ids['btn']
        btn.text = 'Read and solve sudoku'

        # Update label
        lbl = self.ids['lbl']
        lbl.text = ''
        

class TestCamera(App):

    def build(self):
        return CameraClick()


TestCamera().run()