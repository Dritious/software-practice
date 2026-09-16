import numpy as np
from pathlib import Path

class MatrixLab:
    def run(self) -> None:
        arr = self.create_array()
        A = self.array_to_matrix_A(arr)
        A = self.make_linear_operations(A)
        B = self.create_matrix_B()
        self.vectors_sum(A, B)
        self.matrix_product(A, B)
        A, B = self.sqrt_matrix(A, B)
        self.inv_of_matrix(A, B)
        A, B = self.power_of_matrixes(A, B)

        self.solve_slae()

        self.matrix_analyse(A, B)
        self.save_and_load_matrixes(A, B)


    @staticmethod
    def create_array() -> np.ndarray:
        my_array = np.array([x for x in range(10, 70, 2)])
        print(f"1. Array: \n{my_array}\n")
        return my_array

    @staticmethod
    def array_to_matrix_A(arr: np.ndarray) -> np.ndarray:
        matrix = arr.reshape(6, 5)
        matrix_T = matrix.T 
        print(f"2. Matrix A (transposed): \n{matrix_T}\n, Shape: \n{matrix_T.shape}\n")
        return matrix_T

    @staticmethod
    def make_linear_operations(matrix: np.ndarray):
        res_matrix = 2.5 * matrix - 5
        print(f"3. Result of linear operations: \n{res_matrix}\n, Min: \n{res_matrix.min()}\n")
        return res_matrix

    @staticmethod
    def create_matrix_B() -> np.ndarray:
        rng = np.random.default_rng(seed=42)
        matrix = rng.integers(0, 10, size=(6, 3))
        print(f"4. Matrix B: \n{matrix}\n, Shape: \n{matrix.shape}\n")
        return matrix

    @staticmethod
    def vectors_sum(A: np.ndarray, B: np.ndarray) -> (np.ndarray, np.ndarray):
        a = A.sum(axis=1)
        print(f"5. Vector a (sum of rows in A): \n{a}\n, Shape: \n{a.shape}\n")
        b = B.sum(axis=0)
        print(f"Vector b (sum of columns in B): \n{b}\n, Shape: \n{b.shape}\n")
        return a, b

    @staticmethod
    def matrix_product(A: np.ndarray, B: np.ndarray):
        product = A @ B
        print(f"6. Matrix product (A @ B): \n{product}\n")

    @staticmethod
    def sqrt_matrix(A: np.ndarray, B: np.ndarray):
        A = np.delete(A, 2, axis=1)
        rng = np.random.default_rng(seed=42)
        B = np.append(B, rng.integers(10, 20, size=(6, 3)), axis=1)
        print(f"7. Matrix A after sqrt: \n{A}\n, Shape: \n{A.shape}\n")
        print(f"Matrix B after sqrt: \n{B}\n, Shape: \n{B.shape}\n")
        return A, B
    
    @staticmethod
    def inv_of_matrix(A: np.ndarray, B):
        det_A = np.linalg.det(A)
        print(det_A)
        try:
            inv_A = np.linalg.inv(A)
            print(f"8. Inverse of matrix A: \n{inv_A}\n")
        except np.linalg.LinAlgError as e:
            print(f"Обратной матрицы для A не существует: {e}\n")           
        det_B = np.linalg.det(B)
        print(det_B)
        try:
            inv_B = np.linalg.inv(B)
            print(f"Inverse of matrix B: \n{inv_B}\n")
        except np.linalg.LinAlgError as e:
            print(f"Обратной матрицы для B не существует: {e}\n")           
    
    @staticmethod
    def power_of_matrixes(A, B):
        A_6 = np.linalg.matrix_power(A, 6)
        print(f"9. Matrix A 6th power: \n{A_6}\n")
        B_14 = np.linalg.matrix_power(B, 14)
        print(f"Matrix B 14th power: \n{B_14}\n")
        return A, B
    
    @staticmethod
    def solve_slae():
        A = [
            [1, 0, -1, 25],
            [-6, 28, -7.4, 0],
            [1, -5, 13, 2.8],
            [4, 0, 3, 1.7]
            ]
        b = [6.7, -4, 16, 8]
        try:
            res = np.linalg.solve(A,b)
            print(f"10. Solution of the system: \n{res}\n")
        except np.linalg.LinAlgError as e:
            print(f"There is no solution or there is infinitely many solutions: {e}\n")

    @staticmethod
    def matrix_analyse(A, B):
        rank_A = np.linalg.matrix_rank(A)
        rank_B = np.linalg.matrix_rank(B)
        print(f"11. Ranks: \n{rank_A}\n, \n{rank_B}\n")

        mean_A = A.mean()
        mean_B = B.mean()
        print(f"Means: \n{mean_A}\n, \n{mean_B}\n")

        cov_A = np.cov(A)
        cov_B = np.cov(B)
        print(f"Covaration matrixes: \n{cov_A}\n, \n{cov_B}\n")

        row, col = np.unravel_index(A.argmax(), A.shape)
        print(f"Argmax of A: \n{row}\n, \n{col}\n")

        vec_B = B.reshape(36)
        print(f"Vector from B: \n{vec_B}\n")
    
    @staticmethod
    def save_and_load_matrixes(A, B):
        file_A = Path("matrix_A.csv")
        file_B = Path("matrix_B.csv")

        np.savetxt(file_A, A, delimiter=",")
        np.savetxt(file_B, B, delimiter=",")

        loaded_A = np.loadtxt(file_A, delimiter=",")
        loaded_B = np.loadtxt(file_B, delimiter=",")

        diff_A = np.max(np.abs(A - loaded_A))
        diff_B = np.max(np.abs(B - loaded_B))

        print(f"Max diff A: \n{diff_A:.2e}\n")
        print(f"Max diff B: \n{diff_B:.2e}\n")
        print(f"Is successful: \n{diff_A < 1e-6 and diff_B < 1e-6}\n")

if __name__ == "__main__":
    matrix_lab = MatrixLab()
    matrix_lab.run()     

