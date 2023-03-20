import random

def main() -> int:
    print('tell the duck your problems\n')
    kill_seq = ['BYE', 'EXIT', 'EXIT()', 'GO AWAY', 'STOP', 'KYS']
    while(True):
        try:
            i = input()
            if i.upper() in kill_seq: raise KeyboardInterrupt
            if i.upper() == 'QUACK': print('stop copying me', end=' ')
            print('*quack* ' * random.randint(1,3))
        except KeyboardInterrupt:
            print('*Quack*! (Bye!)')
            return 0
    
if __name__ == '__main__':
    main()