from Deck import Deck
import numpy as np
from Card import Card
from Suits import Suits

class BJGame():
    def __init__(self):
        self.deck = self.get_suffled_deck()
        self.dealer_hand = []
        self.player_hand = []
        self.suffle = False

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
        if card.value != -1:
            return card
        else:
            self.suffle = True
            self.get_card() 
    
    def print_hand(self, name, hand):
        print(name,": ",end=" ")
        for card in hand:
            print(card, "|", end=" ")
        print(self.get_hand_value(hand))

    def get_hand_value(self, hand):
        sum_value = 0
        for card in hand:
            sum_value += card.get_value()

        for card in hand:
            if sum_value > 21 and card.value == "A":
                sum_value -= 10

        return sum_value

    def play_round(self, wager):
        self.dealer_hand = [self.get_card(),self.get_card()]
        self.player_hand = [self.get_card(),self.get_card()]
        print("Dealers Hand: ", self.dealer_hand[0], "|", "X , X")
        self.print_hand("Player Hand", self.player_hand)
        
        if self.get_hand_value(self.player_hand) == 21:
            return wager * 2

        
        while True:
            move = input("[H] = Hit [S] = Stand: ")
            if move == "H":
                self.player_hand.append(self.get_card())
            else:
                break
            print("Dealers Hand: ", self.dealer_hand[0], "|", "X , X")
            self.print_hand("Player Hand", self.player_hand)
            if self.get_hand_value(self.player_hand) > 21:
                return -wager
            elif self.get_hand_value(self.player_hand) == 21:
                return wager
        
        self.print_hand("Dealers Hand", self.dealer_hand)
        self.print_hand("Player Hand", self.player_hand)

        while self.get_hand_value(self.dealer_hand) <= 17:
            print("=================================================")
            self.dealer_hand.append(self.get_card())
            self.print_hand("Dealers Hand", self.dealer_hand)
            self.print_hand("Player Hand", self.player_hand)

        if self.get_hand_value(self.dealer_hand) > 21:
            return wager
        elif self.get_hand_value(self.dealer_hand) > self.get_hand_value(self.player_hand):
            return -wager
        elif self.get_hand_value(self.dealer_hand) < self.get_hand_value(self.player_hand):
            return wager
        else:
            return 0

