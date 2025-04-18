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

    def play_round(self, wager, hand):

        self.print_hand("Player Hand", hand)
        
        while True:
            if len(hand) < 3:
                move = input("[H] = Hit [S] = Stand [D] = Double: ")
            else:
                move = input("[H] = Hit [S] = Stand: ")

            if move == "H":
                hand.append(self.get_card())
            elif move == "D":
                hand.append(self.get_card())
                wager *= 2
                break
            else:
                break
            print("Dealers Hand: ", self.dealer_hand[0], "|", "X , X")
            self.print_hand("Player Hand", hand)
            if self.get_hand_value(hand) > 21:
                return -wager
            elif self.get_hand_value(hand) == 21:
                return wager
        
        self.print_hand("Dealers Hand", self.dealer_hand)
        self.print_hand("Player Hand", hand)

        while self.get_hand_value(self.dealer_hand) < 17:
            print("=================================================")
            self.dealer_hand.append(self.get_card())
            self.print_hand("Dealers Hand", self.dealer_hand)
            self.print_hand("Player Hand", hand)

        if self.get_hand_value(self.dealer_hand) > 21:
            return wager
        elif self.get_hand_value(self.dealer_hand) > self.get_hand_value(hand):
            return -wager
        elif self.get_hand_value(self.dealer_hand) < self.get_hand_value(hand):
            return wager
        else:
            return 0
    
    def main(self, wager):
        winings = 0
        self.dealer_hand = [self.get_card(),self.get_card()]
        self.player_hand = [self.get_card(),self.get_card()]
        print("Dealers Hand: ", self.dealer_hand[0], "|", "X , X")
                
        if self.get_hand_value(self.player_hand) == 21:
            self.player_hand = self.split(self.player_hand)
            return wager * 1.5

        if self.player_hand[0].get_value() == self.player_hand[1].get_value():
            self.print_hand("Player Hand", self.player_hand)
            if input("[H] = Hit [S] = Stand [D] = Double [SP] = Split: ") == "SP":
                self.player_hand = self.split(self.player_hand)
        if len(np.asarray(self.player_hand).shape) > 1 :
            for i in range(len(np.asarray(self.player_hand).shape)):
                winings += self.play_round(wager, self.player_hand[i])
        else:
            return self.play_round(wager, self.player_hand)
    
        return winings

    def split(self, hand):
        return [[hand[0], self.get_card()],[hand[1], self.get_card()]]
