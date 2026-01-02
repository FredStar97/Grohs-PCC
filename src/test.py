import numpy as np


def main():
    data = np.load("../data/processed/encoding_ffn.npy")
    print(data)
    try:
        arrays = {k: data[k] for k in data.files}
    finally:
        data.close()

    print(arrays.keys())


if __name__ == "__main__":
    main()
