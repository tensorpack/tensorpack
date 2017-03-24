from case_script import TestPythonScript


class MnistTest(TestPythonScript):

    def setUp(self):
        TestPythonScript.clear_trainlog('../examples/mnist-convnet.py')

    def testScript(self):
        self.assertSurvive('../examples/mnist-convnet.py', args=None, timeout=10)

    def tearDown(self):
        TestPythonScript.clear_trainlog('../examples/mnist-convnet.py')
