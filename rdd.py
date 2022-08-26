import random

def main() -> int:
    print('tell the duck your problems\n')
    while(True):
        try:
            i = input()
            if i.upper() == 'BYE': raise KeyboardInterrupt
            print('*quack* ' * random.randint(1,3))
        except KeyboardInterrupt:
            print('*Quack*! (Bye!)')
            return 0
    
if __name__ == '__main__':
    main()