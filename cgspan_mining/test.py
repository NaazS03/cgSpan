import unittest
from .cgspan import cgSpan

class cgSpanTests(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.f = open("unittestCloseGraphOutput.txt", "w")

    @classmethod
    def tearDownClass(self) -> None:
        self.f.close()

    def updateOutput(self,results):
        self.f.write(str(sorted(results)) + "\n")

    def convert_results_format(self,results):
        results_as_tuples = []
        for result in results:
            support,description,num_vertices = result[0],result[1],result[2]
            results_as_tuples.append((support,description,num_vertices))

        return results_as_tuples

    def test_case_00(self):
        cg = cgSpan(
            database_file_name="../graphdata/unit_test_data/graph.data.testcase.0",
            min_support=2
        )
        cg.run()

        #result = [support, description, num_vert]
        results = cg._report_df.to_numpy().astype(str)
        results = self.convert_results_format(results)

        solution_1 = ('2','v 0 A v 1 B v 2 C v 3 D e 0 1 x e 1 2 y e 1 3 z ','4')
        solutions = [solution_1]

        self.updateOutput(results)
        self.assertEqual(sorted(solutions), sorted(results), "Atleast one solution was not found.")

    def test_case_01(self):
        cg = cgSpan(
            database_file_name="../graphdata/unit_test_data/graph.data.testcase.1",
            min_support=2
        )
        cg.run()

        results = cg._report_df.to_numpy().astype(str)
        results = self.convert_results_format(results)

        solution_1 = ('2','v 0 A v 1 B v 2 C v 3 D e 0 1 x e 1 2 y e 2 3 z ','4')
        solutions = [solution_1]

        self.updateOutput(results)
        self.assertEqual(sorted(solutions), sorted(results), "Atleast one solution was not found.")

    def test_case_02(self):
        cg = cgSpan(
            database_file_name="../graphdata/unit_test_data/graph.data.testcase.2",
            min_support=2
        )
        cg.run()

        results = cg._report_df.to_numpy().astype(str)
        results = self.convert_results_format(results)

        solution_1 = ('2','v 0 A v 1 B v 2 C e 0 1 x e 0 2 z e 1 2 y ','3')
        solutions = [solution_1]

        self.updateOutput(results)
        self.assertEqual(sorted(solutions), sorted(results), "Atleast one solution was not found.")

    def test_case_03(self):
        cg = cgSpan(
            database_file_name="../graphdata/unit_test_data/graph.data.testcase.3",
            min_support=2
        )
        cg.run()

        results = cg._report_df.to_numpy().astype(str)
        results = self.convert_results_format(results)

        solution_1 = ('2','v 0 A v 1 B v 2 C v 3 D v 4 E e 0 1 w e 0 2 x e 0 3 y e 0 4 z ','5')
        solutions = [solution_1]

        self.updateOutput(results)
        self.assertEqual(sorted(solutions), sorted(results), "Atleast one solution was not found.")

    def test_case_04(self):
        cg = cgSpan(
            database_file_name="../graphdata/unit_test_data/graph.data.testcase.4",
            min_support=2
        )
        cg.run()

        results = cg._report_df.to_numpy().astype(str)
        results = self.convert_results_format(results)

        solution_1 = ('2','v 0 A v 1 D v 2 C v 3 B e 0 1 x e 0 2 y e 0 3 z e 1 2 v e 2 3 w ','4')
        solutions = [solution_1]

        self.updateOutput(results)
        self.assertEqual(sorted(solutions), sorted(results), "Atleast one solution was not found.")

    def test_case_05(self):
        cg = cgSpan(
            database_file_name="../graphdata/unit_test_data/graph.data.testcase.5",
            min_support=2
        )
        cg.run()

        results = cg._report_df.to_numpy().astype(str)
        results = self.convert_results_format(results)

        solution_1 = ('2','v 0 A v 1 D v 2 E v 3 B v 4 C e 0 1 y e 0 3 z e 1 2 w e 3 4 x ','5')
        solutions = [solution_1]

        self.updateOutput(results)
        self.assertEqual(sorted(solutions), sorted(results), "Atleast one solution was not found.")

    def test_testcase_06(self):
        #Placeholder test for repeat of testcase 1 with a third graph added
        self.assertEqual(True,True)

    def test_case_07(self):
        cg = cgSpan(
            database_file_name="../graphdata/unit_test_data/graph.data.testcase.7",
            min_support=2
        )
        cg.run()

        results = cg._report_df.to_numpy().astype(str)
        results = self.convert_results_format(results)

        solution_1 = ('2','v 0 A v 1 B v 2 C e 0 1 x e 0 2 z e 1 2 y ','3')
        solution_2 = ('3','v 0 A v 1 B v 2 C e 0 1 x e 1 2 y ','3')
        solutions = [solution_1, solution_2]

        self.updateOutput(results)
        self.assertEqual(sorted(solutions), sorted(results), "Atleast one solution was not found.")

    def test_case_08(self):
        cg = cgSpan(
            database_file_name="../graphdata/unit_test_data/graph.data.testcase.8",
            min_support=2
        )
        cg.run()

        results = cg._report_df.to_numpy().astype(str)
        results = self.convert_results_format(results)

        solution_1 = ('2','v 0 A v 1 B v 2 C v 3 D v 4 E e 0 1 w e 0 2 x e 0 3 y e 0 4 z ','5')
        solution_2 = ('3','v 0 A v 1 B v 2 C v 3 D e 0 1 w e 0 2 x e 0 3 y ','4')
        solutions = [solution_1, solution_2]

        self.updateOutput(results)
        self.assertEqual(sorted(solutions), sorted(results),"Atleast one solution was not found.")

    def test_case_09(self):
        cg = cgSpan(
            database_file_name="../graphdata/unit_test_data/graph.data.testcase.9",
            min_support=2
        )
        cg.run()

        results = cg._report_df.to_numpy().astype(str)
        results = self.convert_results_format(results)

        solution_1 = ('2','v 0 A v 1 D v 2 C v 3 B e 0 1 x e 0 2 y e 0 3 z e 1 2 v e 2 3 w ','4')
        solution_2 = ('3','v 0 A v 1 D v 2 C v 3 B e 0 1 x e 0 2 y e 0 3 z e 2 3 w ','4')
        solutions = [solution_1, solution_2]

        self.updateOutput(results)
        self.assertEqual(sorted(solutions),sorted(results),"Atleast one solution was not found.")

    def test_case_10(self):
        cg = cgSpan(
            database_file_name="../graphdata/unit_test_data/graph.data.testcase.10",
            min_support=2
        )
        cg.run()

        results = cg._report_df.to_numpy().astype(str)
        results = self.convert_results_format(results)

        solution_1 = ('2', 'v 0 A v 1 D v 2 E v 3 B v 4 C e 0 1 y e 0 3 z e 1 2 w e 3 4 x ', '5')
        solution_2 = ('3', 'v 0 A v 1 D v 2 B v 3 C e 0 1 y e 0 2 z e 2 3 x ', '4')
        solutions = [solution_1, solution_2]

        self.updateOutput(results)
        self.assertEqual(sorted(solutions), sorted(results), "Atleast one solution was not found.")

    def test_case_11(self):
        cg = cgSpan(
            database_file_name="../graphdata/unit_test_data/graph.data.testcase.11",
            min_support=2
        )
        cg.run()

        results = cg._report_df.to_numpy().astype(str)
        results = self.convert_results_format(results)

        solution_1 = ('2','v 0 A v 1 AA v 2 B v 3 C v 4 D v 5 E v 6 F v 7 G e 0 1 m e 1 2 n e 1 5 q e 2 3 o e 3 4 p e 5 6 r e 6 7 s ','8')
        solution_2 = ('3','v 0 A v 1 AA v 2 B v 3 E e 0 1 m e 1 2 n e 1 3 q ','4')
        solutions = [solution_1, solution_2]

        self.updateOutput(results)
        self.assertEqual(sorted(solutions), sorted(results), "Atleast one solution was not found.")

if __name__ == '__main__':
    unittest.main()
