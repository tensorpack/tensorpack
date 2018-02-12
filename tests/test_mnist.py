from case_script import TestPythonScript


class MnistTest(TestPythonScript):

    @property
    def script(self):
        return '../examples/basics/mnist-convnet.py'

    def test(self):
        self.assertSurvive(self.script, args=None)
