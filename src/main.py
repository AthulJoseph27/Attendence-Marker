import generateDB
import preprocess
import train


def main():
    generateDB.main()
    preprocess.main()
    train.main()


if __name__ == "__main__":
    main()
