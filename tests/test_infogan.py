from case_script import TestPythonScript


class InfoGANTest(TestPythonScript):

    @property
    def script(self):
        return '../examples/GAN/InfoGAN-mnist.py'

    def test(self):
        self.assertSurvive(self.script, args=None)
