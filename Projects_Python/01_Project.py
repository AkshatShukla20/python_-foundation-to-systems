'''
This is a simple snake water gun game in python and this is our first project 
Let's play the game and have fun with it.
Rules of the game:
1. Snake vs Water: Snake drinks the water, so snake wins.
2. Water vs Gun: Gun will drown in water, so water wins.
3. Gun vs Snake: Gun will kill the snake, so gun wins.

1 For Snake
2 For Water
3 for Gun

'''
from random import randint   # This will import the random module and randint function from it to generate a random number between 1 and 3 for the computer's choice.
random_number = randint(1,3)

You = int(input("Enter your choice: "))
dict = { 1: "Snake" , 2: "Water" , 3: "Gun" }
Computer = dict[random_number]
You = dict[You]

if( You == Computer):
    print("Tie!")
    print(Computer)
elif( You == "Snake" and Computer == "Water"):
    print("You Win!")
    print(Computer)
elif( You == "Water" and Computer == "Snake"):
    print("Computer Wins!")
    print(Computer)
elif( You == "Water" and Computer == "Gun"):
    print("You Win!")
    print(Computer)
elif( You == "Gun" and Computer == "Water"):
    print("Computer Wins!")
    print(Computer)
elif( You == "Snake" and Computer == "Gun"):
    print("Computer Wins!")
    print(Computer)
elif( You == "Gun" and Computer == "Snake"):
    print("You Win!")
    print(Computer)
else:
    print("You Are The Champion!")
    print(Computer)


    '''
    This Project Teaches Us 3 Things:
    1. Use Of Random Module
    2. Use Of If Else Statement
    3. Use Of Dictionary
    
    '''