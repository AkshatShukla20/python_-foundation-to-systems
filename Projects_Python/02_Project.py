# The Guess the Number Game 

import random 

n = random.randint(1, 100)
Guesses = 0

while n != -1:
    User_Guess = int(input("Enter the number between 1 and 100: "))
    Guesses += 1
    if ( User_Guess > n ):
        print("Your guess is too high. Try again.")
    elif ( User_Guess < n ):
        print("Your guess is too low. Try again.")
    else:
        print("Congratulations! You guessed the number.")
        break
print(f"Number of guesses: {Guesses}")