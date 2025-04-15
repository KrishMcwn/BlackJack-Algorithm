from Deck import Deck
import numpy as np
from Card import Card
from Suits import Suits

class BJGame():
    def __init__(self):
        self.deck = self.get_suffled_deck()
        self.dealer_hand = []
        self.player_hand = []

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

    def play_round(self, wager):
        self.dealer_hand = [self.get_card(),self.get_card()]
        self.player_hand = [self.get_card(),self.get_card()]
        print("Dealers Hand:")
        print(self.dealer_hand[0])
        print("players Hand:")
        self.print_hand(self.player_hand)
        print("Hand Value: ",self.get_hand_value(self.player_hand))

        while True:
            move = input("[H] = Hit, [S] = Stand : ")
            if move == "S":
                break
            else:
                self.player_hand.append(self.get_card())
                print("New Player Hand:")
                self.print_hand(self.player_hand)
                print("Hand Value: ",self.get_hand_value(self.player_hand))
                if self.get_hand_value(self.player_hand) > 21:
                    return 0

        while self.get_hand_value(self.dealer_hand) < 17:
            self.dealer_hand.append(self.get_card())
            print("New Dealer Hand:")
            self.print_hand(self.dealer_hand)
            print("Hand Value: ", self.get_hand_value(self.dealer_hand))
            if self.get_hand_value(self.dealer_hand) > 21:
                return wager*2

        print("Dealers Hand:")
        self.print_hand(self.dealer_hand)
        print("Hand Value: ",self.get_hand_value(self.dealer_hand))
        print("players Hand:")
        self.print_hand(self.player_hand)
        print("Hand Value: ",self.get_hand_value(self.player_hand)) 
        if self.get_hand_value(self.dealer_hand) > self.get_hand_value(self.player_hand):
            return 0
        elif self.get_hand_value(self.dealer_hand) < self.get_hand_value(self.player_hand):
            return wager*2
        else:
            return wager
        
