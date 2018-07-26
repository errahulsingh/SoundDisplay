"""
Code for MAX7218 from maxim
Converted from this code: http://playground.arduino.cc/LEDMatrix/Max7219
"""

from time import sleep

from pyfirmata import Arduino

HIGH = 1
LOW = 0

dataIn = 2
load = 4
clock = 3
maxInUse = 1

max7219_reg_noop = 0x00
max7219_reg_digit0 = 0x01
max7219_reg_digit1 = 0x02
max7219_reg_digit2 = 0x03
max7219_reg_digit3 = 0x04
max7219_reg_digit4 = 0x05
max7219_reg_digit5 = 0x06
max7219_reg_digit6 = 0x07
max7219_reg_digit7 = 0x08
max7219_reg_decodeMode = 0x09
max7219_reg_intensity = 0x0a
max7219_reg_scanLimit = 0x0b
max7219_reg_shutdown = 0x0c
max7219_reg_displayTest = 0x0f


class LedMatrix:
    def __init__(self, board):
        self.board = board

    def _digitalWrite(self, pin, val):
        self.board.digital[pin].write(val)

    def putByte(self, data):
        for i in range(8, 0, -1):
            mask = 0x01 << (i - 1)
            self._digitalWrite(clock, LOW)
            if (data & mask):
                self._digitalWrite(dataIn, HIGH)
            else:
                self._digitalWrite(dataIn, LOW)
            self._digitalWrite(clock, HIGH)

    def maxSingle(self, reg, col):
        self._digitalWrite(load, LOW)
        self.putByte(reg)
        self.putByte(col)
        self._digitalWrite(load, LOW)
        self._digitalWrite(load, HIGH)

    def maxAll(self, reg, col):
        self._digitalWrite(load, LOW)
        for _ in range(1, maxInUse + 1):
            self.putByte(reg)
            self.putByte(col)
        self._digitalWrite(load, LOW)
        self._digitalWrite(load, HIGH)

    def maxOne(self, maxNr, reg, col):
        self._digitalWrite(load, LOW)

        for _ in range(maxInUse, maxNr, -1):
            self.putByte(0)
            self.putByte(0)

        self.putByte(reg)
        self.putByte(col)

        for _ in range(maxNr - 1, 0, -1):
            self.putByte(0)
            self.putByte(0)

        self._digitalWrite(load, LOW)
        self._digitalWrite(load, HIGH)

    def clear(self):
        for e in range(1, 9):
            self.maxAll(e, 0)

    def setup(self):
        print('Initializing matrix...')
        self._digitalWrite(13, HIGH)
        self.maxAll(max7219_reg_scanLimit, 0x07)
        self.maxAll(max7219_reg_decodeMode, 0x00)
        self.maxAll(max7219_reg_shutdown, 0x01)
        self.maxAll(max7219_reg_displayTest, 0x00)
        self.clear()
        self.maxAll(max7219_reg_intensity, 0x0f & 0x0f)
        print('Done')

    def draw_matrix(self, point_matrix):
        for c_id, pointlist in enumerate(point_matrix):
            self.maxSingle(c_id+1, int(''.join(str(v) for v in pointlist), 2))

def loop(matrix):
    """ Verify that the functions work. """
    matrix.maxSingle(1, 1)
    matrix.maxSingle(2, 2)
    matrix.maxSingle(3, 4)
    matrix.maxSingle(4, 8)
    matrix.maxSingle(5, 16)
    matrix.maxSingle(6, 32)
    matrix.maxSingle(7, 64)
    matrix.maxSingle(8, 128)
    sleep(.25)
    matrix.clear()
    sleep(.25)
    matrix.maxAll(1, 1)
    matrix.maxAll(2, 3)
    matrix.maxAll(3, 7)
    matrix.maxAll(4, 15)
    matrix.maxAll(5, 31)
    matrix.maxAll(6, 63)
    matrix.maxAll(7, 127)
    matrix.maxAll(8, 255)
    sleep(.25)
    matrix.clear()
    sleep(.25)
    x = [[1, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 0, 0, 1]]
    matrix.draw_matrix(x)
    sleep(.25)
    matrix.clear()
    sleep(.25)


if __name__ == "__main__":
    board = Arduino('COM3')
    matrix = LedMatrix(board)
    matrix.setup()
    while True:
        loop(matrix)
