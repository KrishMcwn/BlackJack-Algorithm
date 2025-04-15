from Suits import Suits
from Card import Card
from Deck import Deck
from BJGame import BJGame

Card1 = Card(Suits.Hearts,"K") 
Card2 = Card(Suits.Hearts,'A')

deck = Deck(2)
bj = BJGame()
print(bj.play_round(100))
