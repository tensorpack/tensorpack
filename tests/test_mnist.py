from case_script import TestPythonScript


class MnistTest(TestPythonScript):

    @property
    def script(self):
        return '../examples/mnist-convnet.py'

    def test(self):
        self.assertSurvive(self.script, args=None, timeout=10)
