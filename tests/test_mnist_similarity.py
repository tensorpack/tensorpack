from case_script import TestPythonScript


class SimilarityLearningTest(TestPythonScript):

    @property
    def script(self):
        return '../examples/SimilarityLearning/mnist-embeddings.py'

    def test(self):
        self.assertSurvive(self.script, args=['--algorithm triplet'], timeout=10)
