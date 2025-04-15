class Card():
    def __init__(self,suit,value):
        self.suit = suit
        self.value = value
    
    def get_value(self):
        if isinstance(self.value,int):
            return self.value
        elif self.value.isnumeric():
            return int(self.value)
        elif self.value in ["J","Q","K"]:
            return 10
        elif self.value == "A":
            return 11
        else:
            return -1

    def __str__(self):
        return "{suit} , {value}".format(suit = self.suit.name, value = self.value)
