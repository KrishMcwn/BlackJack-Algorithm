from Suits import Suits
from Card import Card
import numpy as np


class Deck():
    def __init__(self,number):
        self.deck = self.make_deck(number)


    def make_deck(self, number):
        deck = np.empty([4*number,13],dtype=Card)
        for i in range(number):
            for suit in range(4):
                deck[suit+(i*4),0] = Card(Suits(suit),"A")
                deck[suit+(i*4),10] = Card(Suits(suit),"J")
                deck[suit+(i*4),11] = Card(Suits(suit),"Q")
                deck[suit+(i*4),12] = Card(Suits(suit),"K")
                for value in range(1,10):
                    deck[suit+(i*4),value] = Card(Suits(suit),value)
                
        return deck.flatten()

