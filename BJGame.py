from Deck import Deck
import numpy as np
from Card import Card
from Suits import Suits

class BJGame():
    def __init__(self):
        self.deck = self.get_suffled_deck()
        self.used_cards = []

    def get_suffled_deck(self):
        deck = Deck(4).deck
        np.random.shuffle(deck)
        deck = np.append(deck,Card(Suits.Hearts,-1))
        deck[[156, 208]] = deck[[208, 156]]
        return deck

    def get_card(self):
        card = self.deck[0]
        self.deck[0] = None
        self.deck = np.roll(self.deck,-1)
        return card
    
    def print_hand(self, hand):
        for card in hand:
            print(card)

    def get_hand_value(self, hand):
        sum_value = 0
        for card in hand:
            sum_value += card.get_value()
        return sum_value

    def play_round(self):
        dealer_hand = [self.get_card(),self.get_card()]
        player_hand = [self.get_card(),self.get_card()]
        print("Dealers Hand:")
        self.print_hand(dealer_hand)
        print("Hand Value: ",self.get_hand_value(dealer_hand))
        print("players Hand:")
        self.print_hand(player_hand)
        print("Hand Value: ",self.get_hand_value(player_hand))
        
