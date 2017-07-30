from case_script import TestPythonScript


class ResnetTest(TestPythonScript):
    @property
    def script(self):
        return '../examples/ResNet/imagenet-resnet.py'

    def test(self):
        self.assertSurvive(
            self.script,
            args=['--fake', '--data_format NHWC'], timeout=20)
