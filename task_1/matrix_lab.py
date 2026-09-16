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
        self.save_matrixes(A, B)


    @staticmethod
    def create_array() -> np.ndarray:
        my_array = np.array([x for x in range(10, 70, 2)])
        print(my_array)
        return my_array

    @staticmethod
    def array_to_matrix_A(arr: np.ndarray) -> np.ndarray:
        matrix = arr.reshape(6, 5)
        matrix_T = matrix.T 
        print(matrix_T, matrix_T.shape)
        return matrix_T

    @staticmethod
    def make_linear_operations(matrix: np.ndarray):
        res_matrix = 2.5 * matrix - 5
        print(res_matrix, res_matrix.min())
        return res_matrix

    @staticmethod
    def create_matrix_B() -> np.ndarray:
        rng = np.random.default_rng(seed=42)
        matrix = rng.integers(0, 10, size=(6, 3))
        print(matrix, matrix.shape)
        return matrix

    @staticmethod
    def vectors_sum(A: np.ndarray, B: np.ndarray) -> (np.ndarray, np.ndarray):
        a = A.sum(axis=0)
        print(a.shape, a)
        b = B.sum(axis=1)
        print(b.shape, b)
        return a, b

    @staticmethod
    def matrix_product(A: np.ndarray, B: np.ndarray):
        product = A @ B
        print(product)
    
    @staticmethod
    def sqrt_matrix(A: np.ndarray, B: np.ndarray):
        A = np.delete(A, 2, axis=1)
        rng = np.random.default_rng(seed=42)
        B = np.append(B, rng.integers(1, 20, size=(6, 3)), axis=1)
        print(A.shape, B.shape)
        return A, B
    
    @staticmethod
    def inv_of_matrix(A: np.ndarray, B):
        det_A = np.linalg.det(A)
        print(det_A)
        try:
            inv_A = np.linalg.inv(A)
            print(inv_A)
        except np.linalg.LinAlgError as e:
            print(f"Обратной матрицы для A не существует: {e}")           
        det_B = np.linalg.det(A)
        print(det_A)
        try:
            inv_B = np.linalg.inv(B)
            print(inv_B)
        except np.linalg.LinAlgError as e:
            print(f"Обратной матрицы для B не существует: {e}")           
    
    @staticmethod
    def power_of_matrixes(A, B):
        A_6 = np.linalg.matrix_power(A, 6)
        print(A_6)
        B_14 = np.linalg.matrix_power(A, 14)
        print(B_14)
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
            print(res)
        except np.linalg.LinAlgError as e:
            print(f"Матрица не имеет решений либо их бесконечно много: {e}")

    @staticmethod
    def matrix_analyse(A, B):
        rank_A = np.linalg.matrix_rank(A)
        rank_B = np.linalg.matrix_rank(B)
        print("Ranks: ", rank_A, rank_B)

        mean_A = A.mean()
        mean_B = B.mean()
        print("Means: ", mean_A, mean_B)

        cov_A = np.cov(A)
        cov_B = np.cov(B)
        print("Covaration matrixes: ", cov_A, cov_B)

        argmax_A = A.argmax()
        print("Argmax of A: ", argmax_A)

        vec_B = B.reshape(36)
        print("Vector from B: ", vec_B)
    
    @staticmethod
    def save_matrixes(A, B):
        file_path = Path("data.txt")
        if not file_path.exists():
            file_path.touch()
        with open(file_path, "w") as f:
            np.savetxt(f, A, delimiter=",")
            np.savetxt(f, B, delimiter=",")

if __name__ == "__main__":
    matrix_lab = MatrixLab()
    matrix_lab.run()     

