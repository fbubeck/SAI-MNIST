
import random
import numpy as np


class SampleData():
    def __init__(self, array_length, min_bias, max_bias):
        self.array_length = array_length
        self.x_array = np.empty(self.array_length, dtype=object)
        self.y_array = np.empty(self.array_length, dtype=object)
        self.noise = np.empty(self.array_length, dtype=object)
        self.min_bias = min_bias
        self.max_bias = max_bias
        self.varianz = 0

    def get_Data(self):
        # Befülle x und y Arrays mit random Werten; formula: y= 2*x
        # Erzeuge zusätzlich einen künstlichen Noice mit random Werten zwischen min_bias und max_bias und addiere ihn zu y-Werten
        for x in range(0, self.array_length):
            IntRandom = random.randint(1, self.array_length)
            self.x_array[x] = IntRandom
            self.y_array[x] = IntRandom*2
            self.noise[x] = random.randint(self.min_bias, self.max_bias)
            self.y_array[x] += self.noise[x]

        # Berechne die Varianz des Noice für die Berechnung des MSE
        self.varianz = np.var(self.noise)
        print("Noise: ", self.noise)
        print("Varianz of Noise: ", self.varianz)         

        return self.x_array, self.y_array, self.varianz
