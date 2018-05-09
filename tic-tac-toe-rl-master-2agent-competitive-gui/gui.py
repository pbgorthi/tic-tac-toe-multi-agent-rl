from tkinter import Tk, Button
from tkinter import font
from copy import deepcopy

class GUI:

  def __init__(self, board):
    self.app = Tk()
    self.app.title('TicTacToe')
    self.app.resizable(width=False, height=False)
    self.font = font.Font(family="Helvetica", size=32)
    self.buttons = {}
    for i in range(0, len(board)):
      x = int(i/3)
      y = int(i%3)
      handler = lambda x=x,y=y: self.move(x,y)
      button = Button(self.app, command=handler, font=self.font, width=4, height=2)
      button.grid(row=y, column=x)
      self.buttons[x,y] = button
    handler = lambda: self.reset()
    button = Button(self.app, text='reset', command=handler)
    button.grid(row=3, column=0, columnspan=3, sticky="WE")
    self.update(board, [None, None, None], False)

  def update(self, board, winning_combo, done):
    for i in range(0, len(board)):
      val = board[i]
      if val==0:
        text = 'X'
      elif val==1:
        text = '0'
      else:
        text = '.'
      x = int(i/3)
      y = int(i%3)
      self.buttons[x,y]['text'] = text
      self.buttons[x,y]['disabledforeground'] = 'black'
      if text=='.':
        self.buttons[x,y]['state'] = 'normal'
      else:
        self.buttons[x,y]['state'] = 'disabled'
    if done:
      for pos in winning_combo:
        if pos == None:
          break
        x = int(pos/3)
        y = int(pos%3)
        self.buttons[x,y]['disabledforeground'] = 'red'
      for x,y in self.buttons:
        self.buttons[x,y]['state'] = 'disabled'
    for i in range(0, len(board)):
      x = int(i/3)
      y = int(i%3)
      self.buttons[x,y].update()

  def mainloop(self):
    self.app.mainloop()

if __name__ == '__main__':
  GUI([-1, -1, -1, -1, -1, -1, -1, -1, -1]).mainloop()