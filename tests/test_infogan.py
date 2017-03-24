from case_script import TestPythonScript


class InfoGANTest(TestPythonScript):

    def setUp(self):
        TestPythonScript.clear_trainlog('../examples/GAN/InfoGAN-mnist.py')

    def testScript(self):
        self.assertSurvive('../examples/GAN/InfoGAN-mnist.py', args=None, timeout=10)

    def tearDown(self):
        TestPythonScript.clear_trainlog('../examples/GAN/InfoGAN-mnist.py')
