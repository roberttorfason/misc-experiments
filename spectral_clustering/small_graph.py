import numpy as np

def main():
    x = np.array([[0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0, 0], 
                  [0, 0, 1, 1, 0, 1, 0], 
                  [0, 0, 0, 0, 1, 0, 1], 
                  [0, 0, 0, 0, 0, 1, 0], 
                  ]
                  )

    x_sum = np.sum(x, axis=0)
    x_diag = np.diag(x_sum)
    print(x_diag)
    print(x_sum)

    w, v = np.linalg.eig(x)

    print(sorted(w))
    #print(v)
    w, v = np.linalg.eig(x + x_diag)

    print(sorted(w))


if __name__ == '__main__':
    main()
