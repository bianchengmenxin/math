import numpy as np
from unicodedata import normalize


def func_1():
    # 定义一个矩阵
    matrix_1 = np.array([[1,2,5], [1/2,1,2],[1/5,1/2,1]])
    matrix_2 = np.array([[1,1,3], [1,1,2],[1/3,1/2,1]])
    matrix_3 = np.array([[1,2,3],[1/2,1,2],[1/3,1/2,1]])
    matrix_4 = np.array([[1,3,5],[1/3,1,4],[1/5,1/4,1]])

    # 计算最大特征值
    eigenvalues_1, eigenvectors_1 = np.linalg.eig(matrix_1)
    max_eigenvalue_1 = np.max(eigenvalues_1)
    eigenvalues_2, eigenvectors_2 = np.linalg.eig(matrix_2)
    max_eigenvalue_2 = np.max(eigenvalues_2)
    eigenvalues_3, eigenvectors_3 = np.linalg.eig(matrix_3)
    max_eigenvalue_3 = np.max(eigenvalues_3)
    eigenvalues_4, eigenvectors_4 = np.linalg.eig(matrix_4)
    max_eigenvalue_4 = np.max(eigenvalues_4)

    print("最大特征值：", max_eigenvalue_1)
    print("最大特征值：", max_eigenvalue_2)
    print("最大特征值：", max_eigenvalue_3)
    print("最大特征值：", max_eigenvalue_4)


    max_eigenvalue = np.array([max_eigenvalue_1, max_eigenvalue_2, max_eigenvalue_3, max_eigenvalue_4])

    #判断矩阵的行数
    n = len(matrix_1)
    # 引入一致性指标
    RI = np.array([0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51])
    # 计算一致性指标
    for i in range(len(max_eigenvalue)):
        CI = (max_eigenvalue[i] - n) / (n - 1)

        for i in range(len(RI)):
            if n == i :
                RI_n = RI[i]
                break
        if CI >= RI_n:
            print("矩阵不一致")
        else:
            print("矩阵一致化程度高")

    # 归一化处理
    ## 计算每列的和
    row_sums = np.sum(matrix_1, axis=0)
    normalize_matrix_1 = matrix_1 / row_sums
    print("归一化矩阵：", normalize_matrix_1)


if __name__ == '__main__':
    func_1()
