import unittest
from closegraph import closeGraph
import pandas as pd
import numpy as np

class CloseGraphTests(unittest.TestCase):
    def test_case0(self):
        cg = closeGraph(
            database_file_name="../graphdata/unit_test_data/graph.data.testcase.0",
            min_support=2
        )
        cg.run()

        result = cg._report_df.to_numpy().astype(str)
        #answer = [support, description, num_vert]
        answer = np.array(['2','v 0 A v 1 B v 2 C v 3 D e 0 1 x e 1 2 y e 1 3 z ','4']).reshape((1,-1))

        self.assertEqual(answer[0][0], result[0][0])
        self.assertEqual(answer[0][1], result[0][1])
        self.assertEqual(answer[0][2], result[0][2])

    def test_case1(self):
        cg = closeGraph(
            database_file_name="../graphdata/unit_test_data/graph.data.testcase.1",
            min_support=2
        )
        cg.run()

        result = cg._report_df.to_numpy().astype(str)
        answer = np.array(['2','v 0 A v 1 B v 2 C v 3 D e 0 1 x e 1 2 y e 2 3 z ','4']).reshape((1,-1))

        self.assertEqual(answer[0][0], result[0][0])
        self.assertEqual(answer[0][1], result[0][1])
        self.assertEqual(answer[0][2], result[0][2])

    def test_case2(self):
        cg = closeGraph(
            database_file_name="../graphdata/unit_test_data/graph.data.testcase.2",
            min_support=2
        )
        cg.run()

        result = cg._report_df.to_numpy().astype(str)
        answer = np.array(['2','v 0 A v 1 B v 2 C e 0 1 x e 0 2 z e 1 2 y ','3']).reshape((1,-1))

        self.assertEqual(answer[0][0], result[0][0])
        self.assertEqual(answer[0][1], result[0][1])
        self.assertEqual(answer[0][2], result[0][2])

    def test_case3(self):
        cg = closeGraph(
            database_file_name="../graphdata/unit_test_data/graph.data.testcase.3",
            min_support=2
        )
        cg.run()

        result = cg._report_df.to_numpy().astype(str)
        answer = np.array(['2','v 0 A v 1 B v 2 C v 3 D v 4 E e 0 1 w e 0 2 x e 0 3 y e 0 4 z ','5']).reshape((1,-1))

        self.assertEqual(answer[0][0], result[0][0])
        self.assertEqual(answer[0][1], result[0][1])
        self.assertEqual(answer[0][2], result[0][2])

    def test_case4(self):
        cg = closeGraph(
            database_file_name="../graphdata/unit_test_data/graph.data.testcase.4",
            min_support=2
        )
        cg.run()

        result = cg._report_df.to_numpy().astype(str)
        answer = np.array(['2','v 0 A v 1 B v 2 C v 3 D e 0 1 z e 0 2 y e 0 3 x e 1 2 w e 2 3 v ','4']).reshape((1,-1))

        self.assertEqual(answer[0][0], result[0][0])
        self.assertEqual(answer[0][1], result[0][1])
        self.assertEqual(answer[0][2], result[0][2])

    def test_case5(self):
        cg = closeGraph(
            database_file_name="../graphdata/unit_test_data/graph.data.testcase.5",
            min_support=2
        )
        cg.run()

        result = cg._report_df.to_numpy().astype(str)
        answer = np.array(['2','v 0 A v 1 B v 2 C v 3 D v 4 E e 0 1 z e 0 3 y e 1 2 x e 3 4 w ','5']).reshape((1,-1))

        self.assertEqual(answer[0][0], result[0][0])
        self.assertEqual(answer[0][1], result[0][1])
        self.assertEqual(answer[0][2], result[0][2])

    def test_case7(self):
        cg = closeGraph(
            database_file_name="../graphdata/unit_test_data/graph.data.testcase.7",
            min_support=2
        )
        cg.run()

        results = cg._report_df.to_numpy().astype(str)
        solution_1 = np.array(['2','v 0 A v 1 B v 2 C e 0 1 x e 0 2 z e 1 2 y ','3']).reshape((1,-1))
        solution_2 = np.array(['3','v 0 A v 1 B v 2 C e 0 1 x e 1 2 y ','3']).reshape((1,-1))
        solutions = [solution_1, solution_2]

        for result_1d in results:
            for solution_index in range(0, len(solutions)):
                solution = solutions[solution_index]
                if solution[0][0] == result_1d[0] and solution[0][1] == result_1d[1] and solution[0][2] == result_1d[2]:
                    self.assertTrue(True, "The proposed answer matches an existing solution.")
                    del solutions[solution_index]
                    break

        if len(solutions) != 0:
            self.assertTrue(False, "All possible solutions were not found")


if __name__ == '__main__':
    unittest.main()
