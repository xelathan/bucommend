from models.bucommend import Bucommend

def main():
    bucommend = Bucommend(input_dim=100)
    bucommend.summary()

if __name__ == "__main__":
    main()